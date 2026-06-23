"""ReportExporter 单元测试."""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rockpore.core.report_exporter import ReportExporter, SUPPORTED_FORMATS


class TestReportExporter(unittest.TestCase):
    """测试统一报告导出器."""

    def setUp(self):
        self.exporter = ReportExporter(
            title="测试报告",
            info={"井号": "A1", "深度": "1234.5 m", "岩性": "花岗岩"},
            summary={"总数": 12, "平均": "5.2 mm"},
            headers=["#", "名称", "数值"],
            rows=[
                [1, "颗粒A", 1.5],
                [2, "颗粒B", 2.3],
                [3, "颗粒C", 0.8],
            ],
            notes="测试备注\n第二行",
        )

    def test_supported_formats(self):
        """至少支持 6 种格式."""
        self.assertGreaterEqual(len(SUPPORTED_FORMATS), 6)
        for fmt in ("html", "txt", "pdf", "xlsx", "docx", "csv"):
            self.assertIn(fmt, SUPPORTED_FORMATS)
            self.assertIn("ext", SUPPORTED_FORMATS[fmt])
            self.assertIn("label", SUPPORTED_FORMATS[fmt])

    def test_format_value(self):
        """格式化各种类型值."""
        e = ReportExporter(title="")
        self.assertEqual(e._format_value(1.0), "1")
        self.assertEqual(e._format_value(0.5), "0.5")
        self.assertEqual(e._format_value(1e-7), "1.000e-07")
        self.assertEqual(e._format_value(np.int64(5)), "5")
        self.assertEqual(e._format_value("hello"), "hello")

    def test_build_text(self):
        """纯文本报告生成."""
        text = self.exporter._build_text()
        self.assertIn("测试报告", text)
        self.assertIn("井号", text)
        self.assertIn("A1", text)
        self.assertIn("颗粒A", text)
        self.assertIn("1.5", text)
        self.assertIn("测试备注", text)

    def test_build_html(self):
        """HTML 报告生成 (不嵌图)."""
        html = self.exporter._build_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("测试报告", html)
        self.assertIn("A1", html)
        self.assertIn("<table", html)
        self.assertIn("颗粒A", html)

    def test_build_html_with_image(self):
        """HTML 报告生成 (嵌图)."""
        e = ReportExporter(
            title="带图报告",
            headers=["#"], rows=[[1]],
            annotated_image=np.zeros((100, 200, 3), dtype=np.uint8),  # BGR
        )
        html = e._build_html()
        self.assertIn("data:image/png;base64,", html)

    def test_export_txt(self):
        """导出 TXT."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.txt")
            ok = self.exporter.export("txt", path)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("测试报告", content)

    def test_export_csv(self):
        """导出 CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            ok = self.exporter.export("csv", path)
            self.assertTrue(ok)
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            # 结构: title + info + 空 + 摘要 + 空 + headers + data
            self.assertEqual(rows[0], ["测试报告"])
            # 最后一组是 headers + data
            self.assertEqual(rows[-4], ["#", "名称", "数值"])  # headers
            self.assertEqual(rows[-3], ["1", "颗粒A", "1.5"])  # 第 1 行
            self.assertEqual(rows[-2], ["2", "颗粒B", "2.3"])
            self.assertEqual(rows[-1], ["3", "颗粒C", "0.8"])

    def test_export_html(self):
        """导出 HTML."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.html")
            ok = self.exporter.export("html", path)
            self.assertTrue(ok)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("测试报告", content)

    def test_export_pdf(self):
        """导出 PDF (需要 reportlab)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pdf")
            try:
                ok = self.exporter.export("pdf", path)
                self.assertTrue(ok)
                self.assertGreater(os.path.getsize(path), 1000)
            except ImportError:
                self.skipTest("reportlab 未安装")

    def test_export_xlsx(self):
        """导出 XLSX (需要 openpyxl)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.xlsx")
            try:
                ok = self.exporter.export("xlsx", path)
                self.assertTrue(ok)
                self.assertGreater(os.path.getsize(path), 1000)
            except ImportError:
                self.skipTest("openpyxl 未安装")

    def test_export_docx(self):
        """导出 DOCX (需要 python-docx)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.docx")
            try:
                ok = self.exporter.export("docx", path)
                self.assertTrue(ok)
                self.assertGreater(os.path.getsize(path), 1000)
            except ImportError:
                self.skipTest("python-docx 未安装")

    def test_auto_extension(self):
        """自动补扩展名."""
        with tempfile.TemporaryDirectory() as tmp:
            # 输入无扩展名,自动加
            path_no_ext = os.path.join(tmp, "report")
            ok = self.exporter.export("pdf", path_no_ext)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(path_no_ext + ".pdf"))

    def test_unsupported_format(self):
        """不支持的格式应抛 ValueError."""
        with self.assertRaises(ValueError):
            self.exporter.export("xyz", "/tmp/test.xyz")

    def test_empty_data(self):
        """空数据也能正常导出."""
        e = ReportExporter(title="空报告")
        text = e._build_text()
        self.assertIn("空报告", text)


if __name__ == "__main__":
    unittest.main()