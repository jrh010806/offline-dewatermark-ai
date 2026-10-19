"""Page IO for the prototype."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

from .contract import PageImage

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_pages(path: str | Path, dpi: int = 200) -> list[PageImage]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _pdf_pages(path, dpi)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [_page_from_array(np.asarray(Image.open(path).convert("RGB")),
                                 dpi, path.stem, 0, "image_file")]
    raise ValueError(f"unsupported input {path.suffix}")


def _pdf_pages(path: Path, dpi: int) -> list[PageImage]:
    doc = fitz.open(str(path))
    pages = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)[..., :3].copy()
            pages.append(_page_from_array(arr, dpi, path.stem, page.number, "pdf_page"))
    finally:
        doc.close()
    return pages


def _page_from_array(arr: np.ndarray, dpi: int, doc_id: str, index: int, kind: str) -> PageImage:
    return PageImage(image=arr, dpi=dpi, doc_id=doc_id, page_index=index, source_kind=kind)


def save_png(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
