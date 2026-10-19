"""Build the two PDF shapes the tender mixes together: a flattened/scanned page
(engine B territory) and a page whose watermark is its own transparent object
(engine A territory).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

A4 = fitz.Rect(0, 0, 595, 842)


def _save_temp(img: np.ndarray, path: Path, alpha: np.ndarray | None = None,
               strength: float = 1.0) -> Path:
    if alpha is None:
        Image.fromarray(img).save(path)
    else:
        rgba = np.empty((*alpha.shape, 4), np.uint8)
        rgba[..., :3] = img
        rgba[..., 3] = np.clip(alpha * strength * 255, 0, 255).astype(np.uint8)
        Image.fromarray(rgba).save(path)
    return path


def build(doc_dir: str | Path, out_dir: str | Path, overlay_strength: float = 0.45) -> dict:
    doc_dir, out_dir = Path(doc_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    meta = json.loads((doc_dir / "dataset.json").read_text("utf-8")) if (doc_dir / "dataset.json").exists() else None

    raster = fitz.open()
    structural = fitz.open()
    pages = sorted(doc_dir.glob("page*_gt.npz"))
    for npz_path in pages:
        gt = np.load(npz_path)
        clean, alpha, color = gt["clean"], gt["alpha"], gt["color"]
        watermarked_path = npz_path.with_name(npz_path.name.replace("_gt.npz", ".png"))
        watermarked = np.asarray(Image.open(watermarked_path).convert("RGB"))

        page = raster.new_page(width=A4.width, height=A4.height)
        _save_temp(watermarked, tmp / "r.png")
        page.insert_image(page.rect, filename=str(tmp / "r.png"))

        sp = structural.new_page(width=A4.width, height=A4.height)
        _save_temp(clean, tmp / "c.png")
        sp.insert_image(sp.rect, filename=str(tmp / "c.png"))
        _save_temp(color.astype(np.uint8).reshape(1, 1, 3), tmp / "w.png",
                   alpha=alpha, strength=overlay_strength)
        sp.insert_image(sp.rect, filename=str(tmp / "w.png"), keep_proportion=False)
    raster.save(out_dir / "raster_flattened.pdf", garbage=4, deflate=True)
    structural.save(out_dir / "structural_overlay.pdf", garbage=4, deflate=True)
    raster.close()
    structural.close()
    return {"pages": len(pages), "meta": bool(meta), "out": str(out_dir)}
