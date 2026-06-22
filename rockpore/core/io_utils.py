"""跨平台图像 I/O 工具.

解决 Windows 下 cv2.imread 无法读取 UTF-8 中文路径的问题
(cv::findDecoder 报 can't open/read file: check file path/integrity).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像,支持中文路径(Windows).
    cv2.imread 在 Windows 下使用 GBK 解码路径,对 UTF-8 中文路径会失败.
    本函数先用 np.fromfile 读取字节,再用 cv2.imdecode 解码,绕过文件名解码.
    """
    path = str(path)
    # 先尝试标准 imread
    img = cv2.imread(path, flags)
    if img is not None:
        return img
    # 备选:字节读取 + imdecode
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Union[str, Path], image: np.ndarray) -> bool:
    """保存图像,支持中文路径."""
    path = str(path)
    try:
        # 先尝试标准 imwrite
        if cv2.imwrite(path, image):
            return True
    except Exception:
        pass
    # 备选:imencode + 字节写入
    try:
        ext = Path(path).suffix.lower().lstrip(".") or "png"
        ok, buf = cv2.imencode(f".{ext}", image)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def find_sample_image() -> Optional[str]:
    """查找示例孔洞图像(支持中文文件名).
    查找顺序:
    1. 当前工作目录下的 孔洞.png
    2. 软件安装目录下的 samples/孔洞.png
    3. 用户主目录下的 孔洞.png
    """
    from pathlib import Path
    candidates = [
        Path.cwd() / "孔洞.png",
        Path.cwd() / "samples" / "孔洞.png",
        Path(__file__).resolve().parent.parent.parent / "孔洞.png",
        Path(__file__).resolve().parent.parent / "samples" / "孔洞.png",
        Path.home() / "孔洞.png",
        Path.home() / "Desktop" / "孔洞.png",
        Path.home() / "Desktop" / "new-koushui" / "孔洞.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


__all__ = ["imread_unicode", "imwrite_unicode", "find_sample_image"]
