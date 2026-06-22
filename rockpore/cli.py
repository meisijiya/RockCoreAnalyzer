"""岩心孔洞分析命令行工具.

用法:
    rockpore-cli analyze <image> [options]
    rockpore-cli accuracy <image> <gt_mask> [options]
    rockpore-cli batch <input_dir> [options]
    rockpore-cli synth [--output PATH]  生成合成测试图

示例:
    rockpore-cli analyze lf.jpg --output report.html
    rockpore-cli accuracy synthetic.png synthetic_gt.png
    rockpore-cli batch ./images --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# 确保能找到 rockpore 包(从项目根目录运行)
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from rockpore.core import (
    analyze_pores, scale_from_dpi, get_image_dpi, extract_pores,
    remove_noise, fill_holes, dilate_region, erode_region,
    detect_pores_robust, evaluate_accuracy, generate_report, ReportData,
    generate_synthetic_rock, SyntheticPore, make_default_synthetic,
    AccuracyReport, imread_unicode, imwrite_unicode,
)
from rockpore.core.calibration import ScaleUnit
from rockpore.core.report import _make_annotated_image


def _print_banner(title: str):
    """打印分节标题."""
    bar = "=" * 60
    print(f"\n{bar}\n {title}\n{bar}")


def cmd_analyze(args: argparse.Namespace) -> int:
    """分析单张岩心图."""
    if not os.path.exists(args.image):
        print(f"错误: 找不到图片 {args.image}", file=sys.stderr)
        return 1
    img = imread_unicode(args.image)
    if img is None:
        print(f"错误: 无法读取图片 {args.image}", file=sys.stderr)
        return 1
    dpi = args.dpi if args.dpi > 0 else get_image_dpi(args.image)
    if args.microscopic:
        scale = scale_from_dpi(dpi, microscopic=True)
    else:
        scale = scale_from_dpi(dpi)
    mask, result = detect_pores_robust(img, scale, min_diameter_real=args.min_diameter)
    _print_banner("岩心孔洞分析结果")
    print(f"图片: {args.image}")
    print(f"尺寸: {img.shape[1]} × {img.shape[0]} 像素")
    print(f"DPI: {dpi}, 标尺: {scale.pixels_per_unit:.3f} 像素/{scale.unit.value}")
    print(f"\n--- 关键指标 ---")
    print(f"孔洞总个数: {result.pore_count}")
    print(f"报告级孔洞数(≥{args.min_diameter:.0f}mm): {result.pore_count_report}")
    print(f"孔洞总面积: {result.total_pore_area_real:.3f} mm²")
    print(f"平均孔洞面积: {result.average_area_real:.3f} mm²")
    print(f"孔洞面孔率: {result.porosity * 100:.2f}%")
    print(f"平均等效直径: {result.average_diameter_real:.2f} mm")
    print(f"最大直径: {result.max_diameter_real:.2f} mm")
    print(f"最小直径: {result.min_diameter_real:.2f} mm")
    print(f"\n--- 大小分类 ---")
    for cls, cnt in result.size_distribution.items():
        pct = cnt / max(1, result.pore_count) * 100
        print(f"  {cls}: {cnt} 个 ({pct:.1f}%)")
    # 详细孔洞表
    if args.verbose and result.pores:
        print(f"\n--- 孔洞详细参数 ---")
        print(f"{'ID':>4} {'面积mm²':>10} {'直径mm':>10} {'分类':>10} {'质心':>14}")
        for p in result.pores[:20]:
            print(f"{p.id:>4} {p.area_real:>10.3f} {p.diameter_real:>10.3f} {p.size_class:>10} "
                  f"({p.centroid[0]:>5.0f},{p.centroid[1]:>5.0f})")
        if len(result.pores) > 20:
            print(f"... 共 {len(result.pores)} 个孔洞,只显示前 20 个")
    # 保存结果
    if args.output:
        out_path = args.output
        annot = _make_annotated_image(img, mask, result)
        if out_path.endswith(".html"):
            data = ReportData(
                project_name=args.project or "岩心孔洞分析",
                sample_id=args.sample_id or Path(args.image).stem,
                analyst=args.analyst or "",
                image_path=args.image,
                image_size=(img.shape[1], img.shape[0]),
                scale_info=f"{scale.pixels_per_unit:.3f} px/{scale.unit.value} (DPI={dpi})",
                remarks=args.remarks or "",
                analysis_result=result,
                original_image=img,
                annotated_image=annot,
            )
            from rockpore.core.report import save_report
            save_report(data, out_path)
            print(f"\n✓ HTML 报告已保存: {out_path}")
        elif out_path.endswith(".json"):
            d = result.to_dict()
            d["image"] = args.image
            d["dpi"] = dpi
            d["scale_pixels_per_unit"] = scale.pixels_per_unit
            d["scale_unit"] = scale.unit.value
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            print(f"\n✓ JSON 结果已保存: {out_path}")
        else:
            imwrite_unicode(out_path, annot)
            print(f"\n✓ 标注图已保存: {out_path}")
    if args.save_mask:
        mask_path = args.save_mask
        imwrite_unicode(mask_path, mask)
        print(f"✓ 掩码已保存: {mask_path}")
    return 0


def cmd_accuracy(args: argparse.Namespace) -> int:
    """评估准确率."""
    img = imread_unicode(args.image)
    gt = imread_unicode(args.gt_mask, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图片 {args.image}", file=sys.stderr)
        return 1
    if gt is None:
        print(f"错误: 无法读取 ground truth {args.gt_mask}", file=sys.stderr)
        return 1
    scale = scale_from_dpi(args.dpi if args.dpi > 0 else 96)
    report = evaluate_accuracy(img, gt, scale, use_enhancement=args.enhance)
    _print_banner("准确率评估报告")
    print(f"图片: {args.image}")
    print(f"Ground Truth: {args.gt_mask}")
    print(f"\n--- 像素级 ---")
    print(f"  IoU: {report.pixel_iou * 100:.2f}%")
    print(f"  Precision: {report.pixel_precision * 100:.2f}%")
    print(f"  Recall: {report.pixel_recall * 100:.2f}%")
    print(f"\n--- 检测级(IoU阈值={report.iou_threshold}) ---")
    print(f"  Precision: {report.detection_precision * 100:.2f}%")
    print(f"  Recall: {report.detection_recall * 100:.2f}%")
    print(f"  F1: {report.detection_f1 * 100:.2f}%")
    print(f"\n--- 直径估计 ---")
    print(f"  MAE: {report.diameter_mae:.3f} mm")
    print(f"  相对 MAE: {report.diameter_relative_mae * 100:.2f}%")
    print(f"\n--- 数量统计 ---")
    print(f"  GT 总孔洞: {report.pore_count_gt} (报告级 {report.pore_count_gt_report})")
    print(f"  Pred 总孔洞: {report.pore_count_pred} (报告级 {report.pore_count_pred_report})")
    print(f"  正确匹配: {report.matched_pores}")
    print(f"  漏检: {report.missed_pores}")
    print(f"  误检: {report.false_positives}")
    print(f"\n--- 综合评分 ---")
    print(f"  Composite Score: {report.composite_score}")
    target = 0.80
    passed = report.passes_target(target)
    print(f"  目标(≥{target * 100:.0f}%): {'✓ PASS' if passed else '✗ FAIL'}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n✓ 评估结果已保存: {args.output}")
    return 0 if passed else 2


def cmd_batch(args: argparse.Namespace) -> int:
    """批量分析目录下所有图片."""
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"错误: 不是有效目录 {args.input_dir}", file=sys.stderr)
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    if not files:
        print(f"未在 {args.input_dir} 找到图片文件")
        return 1
    print(f"找到 {len(files)} 张图片,开始批量分析...\n")
    summary = []
    for p in files:
        print(f"--- {p.name} ---")
        img = imread_unicode(str(p))
        if img is None:
            print(f"  跳过: 无法读取")
            continue
        dpi = get_image_dpi(str(p))
        scale = scale_from_dpi(dpi)
        mask, result = detect_pores_robust(img, scale, min_diameter_real=2.0)
        print(f"  孔洞数: {result.pore_count} (报告级 {result.pore_count_report}), "
              f"面孔率: {result.porosity * 100:.2f}%, 平均直径: {result.average_diameter_real:.2f}mm")
        # 保存报告
        annot = _make_annotated_image(img, mask, result)
        data = ReportData(
            project_name="批量分析",
            sample_id=p.stem,
            analyst="",
            image_path=str(p),
            image_size=(img.shape[1], img.shape[0]),
            scale_info=f"{scale.pixels_per_unit:.3f} px/{scale.unit.value} (DPI={dpi})",
            analysis_result=result,
            original_image=img,
            annotated_image=annot,
        )
        from rockpore.core.report import save_report
        html_path = out_dir / f"{p.stem}_report.html"
        save_report(data, str(html_path))
        # 保存 JSON
        json_path = out_dir / f"{p.stem}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        # 保存标注图
        annot_path = out_dir / f"{p.stem}_annot.png"
        imwrite_unicode(str(annot_path), annot)
        summary.append({
            "file": p.name,
            "pore_count": result.pore_count,
            "pore_count_report": result.pore_count_report,
            "porosity": result.porosity,
            "avg_diameter_mm": result.average_diameter_real,
        })
    # 汇总
    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 批量分析完成,共 {len(summary)} 张")
    print(f"  汇总: {summary_path}")
    print(f"  报告目录: {out_dir}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """生成合成测试图."""
    output = args.output or "synthetic.png"
    pores, img, gt = make_default_synthetic()
    imwrite_unicode(output, img)
    gt_path = output.rsplit(".", 1)[0] + "_gt.png"
    imwrite_unicode(gt_path, gt)
    print(f"✓ 合成图: {output} ({img.shape[1]}×{img.shape[0]})")
    print(f"✓ Ground truth: {gt_path}")
    print(f"  孔洞数: {len(pores)}")
    if args.accuracy:
        scale = scale_from_dpi(96)
        report = evaluate_accuracy(img, gt, scale)
        print(f"\n准确率自检:")
        print(f"  IoU: {report.pixel_iou * 100:.2f}%")
        print(f"  F1: {report.detection_f1 * 100:.2f}%")
        print(f"  Composite: {report.composite_score}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rockpore-cli",
        description="岩心孔洞分析命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    # analyze
    pa = sub.add_parser("analyze", help="分析单张岩心图")
    pa.add_argument("image", help="输入图片路径")
    pa.add_argument("--dpi", type=int, default=0, help="DPI,默认从图片读取")
    pa.add_argument("--microscopic", action="store_true", help="微观分析(微米单位)")
    pa.add_argument("--min-diameter", type=float, default=2.0, help="报告级最小直径(mm)")
    pa.add_argument("--output", "-o", default="", help="输出文件(.html/.json/.png)")
    pa.add_argument("--save-mask", default="", help="保存掩码到该路径")
    pa.add_argument("--project", default="", help="项目名称")
    pa.add_argument("--sample-id", default="", help="样品编号")
    pa.add_argument("--analyst", default="", help="分析人员")
    pa.add_argument("--remarks", default="", help="备注")
    pa.add_argument("--verbose", "-v", action="store_true", help="显示孔洞详情")
    pa.set_defaults(func=cmd_analyze)
    # accuracy
    pac = sub.add_parser("accuracy", help="评估准确率(需要 ground truth)")
    pac.add_argument("image", help="输入图片")
    pac.add_argument("gt_mask", help="Ground truth 掩码")
    pac.add_argument("--dpi", type=int, default=96, help="DPI")
    pac.add_argument("--enhance", action="store_true", help="使用增强流水线")
    pac.add_argument("--output", "-o", default="", help="保存评估结果为 JSON")
    pac.set_defaults(func=cmd_accuracy)
    # batch
    pb = sub.add_parser("batch", help="批量分析目录下所有图片")
    pb.add_argument("input_dir", help="输入图片目录")
    pb.add_argument("--output-dir", "-o", default="./output", help="输出目录")
    pb.set_defaults(func=cmd_batch)
    # synth
    ps = sub.add_parser("synth", help="生成合成测试图")
    ps.add_argument("--output", "-o", default="synthetic.png", help="输出路径")
    ps.add_argument("--accuracy", action="store_true", help="自动评估准确率")
    ps.set_defaults(func=cmd_synth)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
