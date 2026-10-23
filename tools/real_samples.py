"""Watermarked test samples on REAL base material (PDF pages + photos).

wm/synth.py generates synthetic documents; this tool instead takes the user's
actual files, stamps them with the four pipeline watermark styles, and ships
the ground truth (clean page + alpha mask + veil parameters) next to every
sample so PSNR/SSIM and mask precision are directly computable.

Usage:
    .venv/bin/python tools/real_samples.py \
        --pdf /path/to/20250227.pdf --images-dir /path/to/photos \
        --out data/real_samples --per-style 4
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm.documents import load_pages  # noqa: E402
from wm.synth import WATERMARK_STYLES, apply_watermark, make_watermark  # noqa: E402

# The four styles the tender matrix asks for; corner_id stays available via --styles.
DEFAULT_STYLES = ("tiled_text", "big_center", "dot_grid", "heavy_dark")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_bases(pdf: Path | None, images_dir: Path | None, dpi: int) -> list[dict]:
    bases: list[dict] = []
    if pdf:
        for page in load_pages(pdf, dpi=dpi):
            bases.append({
                "id": f"{page.doc_id}_p{page.page_index + 1:02d}",
                "kind": "pdf_page",
                "image": page.image,
            })
    if images_dir:
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                bases.append({
                    "id": f.stem,
                    "kind": "photo",
                    "image": np.asarray(Image.open(f).convert("RGB")),
                })
    if not bases:
        raise SystemExit("没有可用的底素材：--pdf 与 --images-dir 至少给一个")
    return bases


def save_pdf(pages: list[Path], out_pdf: Path) -> None:
    first, *rest = [Image.open(p) for p in pages]
    first.save(out_pdf, "PDF", save_all=True,
               append_images=rest, resolution=200.0)


def _write_eval_page(doc_dir: Path, n: int, base: dict, style: str,
                     dpi: int, jpeg_q: int, rng: random.Random) -> None:
    clean = base["image"]
    alpha, color, strength = make_watermark(clean.shape, style, dpi, n, rng)
    wm_img = apply_watermark(clean, alpha, color, strength)
    # 与 jpg/ 交付件同一份像素（JPEG 往返损伤），评测口径才对得上
    buf = io.BytesIO()
    Image.fromarray(wm_img).save(buf, "JPEG", quality=jpeg_q)
    Image.open(buf).convert("RGB").save(doc_dir / f"page{n:02d}.png")
    np.savez_compressed(doc_dir / f"page{n:02d}_gt.npz",
                        clean=clean, alpha=alpha, color=color,
                        strength=np.float32(strength))


def export_eval_grouped(bases: list[dict], styles: tuple[str, ...], eval_out: Path,
                        dpi: int, jpeg_q: int, seed: int) -> None:
    """wm eval 目录格式，但按底素材类型分组：
    documents/ 同一文档多页共享恒定水印（跨页先验可生效）；photos/ 每张单页成档。"""
    doc_bases = [b for b in bases if b["kind"] == "pdf_page"]
    photo_bases = [b for b in bases if b["kind"] == "photo"]
    for si, style in enumerate(styles):
        if doc_bases:
            d = eval_out / "documents" / f"doc{si:02d}_{style}"
            d.mkdir(parents=True, exist_ok=True)
            rng = random.Random(seed + 1000 + si)
            for n, base in enumerate(doc_bases):
                _write_eval_page(d, n, base, style, dpi, jpeg_q, rng)
        for k in range(2):
            if not photo_bases:
                break
            base = photo_bases[(si * 2 + k) % len(photo_bases)]
            p = eval_out / "photos" / f"doc{si * 2 + k:02d}_{style}"
            p.mkdir(parents=True, exist_ok=True)
            _write_eval_page(p, 0, base, style, dpi, jpeg_q,
                             random.Random(seed + 2000 + si * 2 + k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, help="底素材 PDF（逐页渲染为图片）")
    ap.add_argument("--images-dir", type=Path, help="底素材图片目录")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--eval-out", type=Path,
                    help="额外导出 wm eval 目录格式（docNN_style/pageNN_gt.npz + pageNN.png）")
    ap.add_argument("--per-style", type=int, default=4, help="每种水印样式的样本数（3-5）")
    ap.add_argument("--styles", default=",".join(DEFAULT_STYLES),
                    help=f"逗号分隔样式，可选：{','.join(WATERMARK_STYLES)}")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--jpeg-q", type=int, default=92, help="JPG 样本的压缩质量")
    args = ap.parse_args()

    if not 1 <= args.per_style <= 10:
        raise SystemExit("--per-style 建议在 3-5 之间")

    bases = collect_bases(args.pdf, args.images_dir, args.dpi)
    styles = tuple(s.strip() for s in args.styles.split(",") if s.strip())
    for s in styles:
        if s not in WATERMARK_STYLES:
            raise SystemExit(f"未知样式 {s}，可选：{WATERMARK_STYLES}")

    out = args.out
    for sub in ("clean", "jpg", "mask", "pdf"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "seed": args.seed, "dpi": args.dpi, "jpeg_q": args.jpeg_q,
        "styles": list(styles), "per_style": args.per_style,
        "note": "底素材为真实文档/图片；mask 为水印 alpha 覆盖真值，clean 为无水印原图",
        "samples": [],
    }

    rng = random.Random(args.seed)
    for si, style in enumerate(styles):
        pdf_pages: list[Path] = []
        for n in range(args.per_style):
            base = bases[(si * args.per_style + n) % len(bases)]
            clean = base["image"]
            alpha, color, strength = make_watermark(clean.shape, style, args.dpi, n, rng)
            wm_img = apply_watermark(clean, alpha, color, strength)

            sid = f"{style}_{n + 1:02d}"
            clean_p = out / "clean" / f"{sid}.png"
            jpg_p = out / "jpg" / f"{sid}.jpg"
            mask_p = out / "mask" / f"{sid}.png"
            Image.fromarray(clean).save(clean_p)
            Image.fromarray(wm_img).save(jpg_p, "JPEG", quality=args.jpeg_q)
            Image.fromarray((alpha * 255).astype(np.uint8)).save(mask_p)
            np.savez_compressed(out / "mask" / f"{sid}_veil.npz",
                                alpha=alpha, color=color,
                                strength=np.float32(strength))
            pdf_pages.append(jpg_p)
            manifest["samples"].append({
                "id": sid, "style": style, "base": base["id"], "base_kind": base["kind"],
                "jpg": str(jpg_p.relative_to(out)), "clean": str(clean_p.relative_to(out)),
                "mask": str(mask_p.relative_to(out)),
                "strength": round(float(strength), 3),
                "color": [round(float(c), 1) for c in color],
            })
        out_pdf = out / "pdf" / f"{style}.pdf"
        save_pdf(pdf_pages, out_pdf)
        manifest.setdefault("style_pdfs", {})[style] = str(out_pdf.relative_to(out))
        print(f"[{style}] {args.per_style} 张样本 + {out_pdf.name} 完成")

    if args.eval_out:
        export_eval_grouped(bases, styles, args.eval_out, args.dpi,
                            args.jpeg_q, args.seed)
        (args.eval_out / "dataset.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        print(f"eval 格式 -> {args.eval_out}（documents/ 与 photos/ 两组）")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n共 {len(styles) * args.per_style} 组样本，输出目录：{out}")


if __name__ == "__main__":
    main()
