"""Page IO for the prototype."""
from __future__ import annotations

from collections.abc import Iterator
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


def iter_pages(path: str | Path, dpi: int = 200) -> Iterator[PageImage]:
    """Yield pages one at a time — keeps memory flat for large PDFs."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        yield from _iter_pdf(path, dpi)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        yield _page_from_array(np.asarray(Image.open(path).convert("RGB")),
                               dpi, path.stem, 0, "image_file")
    else:
        raise ValueError(f"unsupported input {path.suffix}")


def load_many(paths: list[str | Path], dpi: int = 200) -> list[PageImage]:
    """A hand-picked set of image files is one batch: same watermark, many
    pages, so the cross-page prior still applies."""
    pages: list[PageImage] = []
    for p in paths:
        pages.extend(load_pages(p, dpi=dpi))
    for i, page in enumerate(pages):
        page.page_index = i
    return pages


def count_pages(path: str | Path) -> int:
    """Cheap page count without rasterising — used by the UI progress bar."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        n = len(doc)
        doc.close()
        return n
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    return 0


def _pdf_pages(path: Path, dpi: int) -> list[PageImage]:
    doc = fitz.open(str(path))
    pages = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)[..., :3].copy()
            pages.append(_pdf_page(arr, dpi, path, page.number))
    finally:
        doc.close()
    return pages


def _iter_pdf(path: Path, dpi: int) -> Iterator[PageImage]:
    doc = fitz.open(str(path))
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)[..., :3].copy()
            yield _pdf_page(arr, dpi, path, page.number)
    finally:
        doc.close()


def _pdf_page(arr: np.ndarray, dpi: int, path: Path, index: int) -> PageImage:
    stat = path.stat()
    return PageImage(arr, dpi, path.stem, index, "pdf_page", str(path.resolve()),
                     index, (stat.st_mtime_ns, stat.st_size))


def _page_from_array(arr: np.ndarray, dpi: int, doc_id: str, index: int, kind: str) -> PageImage:
    return PageImage(image=arr, dpi=dpi, doc_id=doc_id, page_index=index, source_kind=kind)


def save_png(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
