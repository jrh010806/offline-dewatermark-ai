"""End-to-end test samples — no CJK font needed.

Generates a set of synthetic document pages with various watermark styles,
runs the full pipeline on each, and reports metrics. Pure OpenCV rendering,
no font dependency.

Usage::

    python -m wm.e2e_test              # generate + evaluate
    python -m wm.e2e_test --out out/e2e  # custom output directory
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .contract import PageImage, Thresholds
from . import pipeline


def _clean_page(h: int = 1200, w: int = 900, seed: int = 0) -> np.ndarray:
    """A realistic document page: white paper, text lines, table, photo block, seal."""
    rng = np.random.RandomState(seed)
    page = np.full((h, w, 3), 242, dtype=np.uint8)

    # Title bar
    cv2.rectangle(page, (60, 40), (w - 60, 100), (30, 30, 30), 3)
    cv2.putText(page, "DOCUMENT", (w // 2 - 120, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3)

    # Body text lines
    y = 140
    for i in range(18):
        length = rng.randint(300, w - 120)
        thickness = rng.choice([1, 2])
        cv2.line(page, (60, y), (60 + length, y), (50, 50, 50), thickness)
        y += rng.randint(28, 38)

    # Table
    table_y, table_h = 300, 200
    table_x, table_w = 80, w - 160
    cv2.rectangle(page, (table_x, table_y), (table_x + table_w, table_y + table_h), (60, 60, 60), 2)
    for row in range(1, 5):
        ry = table_y + row * (table_h // 5)
        cv2.line(page, (table_x, ry), (table_x + table_w, ry), (120, 120, 120), 1)
    for col in range(1, 4):
        cx = table_x + col * (table_w // 4)
        cv2.line(page, (cx, table_y), (cx, table_y + table_h), (120, 120, 120), 1)

    # Photo block (gradient — looks like a photo, must not be removed)
    photo_y = 560
    for dy in range(150):
        v = int(80 + 120 * (dy / 150))
        cv2.line(page, (80, photo_y + dy), (300, photo_y + dy), (v, v - 20, v - 40), 1)
    cv2.rectangle(page, (80, photo_y), (300, photo_y + 150), (40, 40, 40), 2)

    # Dot pattern (must not be confused with watermark dots)
    for dy in range(0, 120, 15):
        for dx in range(0, 200, 15):
            cv2.circle(page, (400 + dx, 580 + dy), 2, (100, 100, 100), -1)

    # Red seal stamp
    cv2.circle(page, (w - 160, h - 160), 60, (30, 30, 200), 3)
    cv2.putText(page, "APPROVED", (w - 220, h - 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 200), 2)

    # Page number
    cv2.putText(page, f"Page {seed + 1}", (w // 2 - 40, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    return page


def _apply_watermark(page: np.ndarray, style: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a watermark. Returns (watermarked, alpha, color)."""
    h, w = page.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float32)
    color = np.array([180, 180, 200], dtype=np.float32)

    if style == "tiled_text":
        for y_off in range(-w, h + w, 100):
            overlay = np.zeros_like(page)
            cv2.putText(overlay, "CONFIDENTIAL", (20, y_off + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            diff = overlay.mean(2)
            alpha = np.maximum(alpha, (diff > 5).astype(np.float32) * 0.35)

    elif style == "big_center":
        overlay = np.zeros_like(page)
        cv2.putText(overlay, "DRAFT", (w // 2 - 180, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 6)
        diff = overlay.mean(2)
        alpha = np.maximum(alpha, (diff > 5).astype(np.float32) * 0.30)

    elif style == "dot_grid":
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        for y in range(20, h, 40):
            for x in range(20, w, 40):
                cv2.circle(mask_u8, (x, y), 5, 102, -1)
        alpha = mask_u8.astype(np.float32) / 255.0 * 0.40
        color = np.array([170, 170, 190], dtype=np.float32)

    elif style == "diagonal_stripe":
        for offset in range(-h, w + h, 60):
            pts = np.array([[offset, 0], [offset + 30, 0],
                            [offset + 30 - h, h], [offset - h, h]], np.int32)
            overlay = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(overlay, [pts], 255)
            alpha = np.maximum(alpha, (overlay > 0).astype(np.float32) * 0.25)

    elif style == "corner_id":
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        for corner_y, corner_x in [(30, 30), (30, w - 200), (h - 60, 30), (h - 60, w - 200)]:
            cv2.putText(mask_u8, "ID-007", (corner_x, corner_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 115, 2)
        alpha = (mask_u8.astype(np.float32) / 255.0) * 0.45

    else:
        raise ValueError(f"unknown style: {style}")

    alpha_3 = alpha[..., None]
    watermarked = (page.astype(np.float32) * (1 - alpha_3) + color * alpha_3).clip(0, 255).astype(np.uint8)
    return watermarked, alpha, color


STYLES = ["tiled_text", "big_center", "dot_grid", "diagonal_stripe", "corner_id"]


def generate(out_dir: Path) -> list[dict]:
    """Generate test pages, one per watermark style. Returns manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, style in enumerate(STYLES):
        clean = _clean_page(seed=i)
        watermarked, alpha, color = _apply_watermark(clean, style, seed=i)

        clean_path = out_dir / f"{i:02d}_{style}_clean.png"
        wm_path = out_dir / f"{i:02d}_{style}_watermarked.png"
        alpha_path = out_dir / f"{i:02d}_{style}_alpha.png"

        cv2.imwrite(str(clean_path), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(wm_path), cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(alpha_path), (alpha * 255).astype(np.uint8))

        manifest.append({
            "style": style,
            "clean": str(clean_path),
            "watermarked": str(wm_path),
            "alpha": str(alpha_path),
        })
    return manifest


def evaluate(manifest: list[dict], out_dir: Path) -> list[dict]:
    """Run the pipeline on each watermarked page and compare against clean."""
    thr = Thresholds()
    results = []
    for entry in manifest:
        style = entry["style"]
        wm_img = cv2.imread(entry["watermarked"])
        clean_img = cv2.imread(entry["clean"])
        alpha_gt = cv2.imread(entry["alpha"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        wm_rgb = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        page = PageImage(image=wm_rgb, dpi=200, doc_id=style, page_index=0)
        t0 = time.perf_counter()
        page_results = pipeline.process_pages([page], thr, force_single=True)
        elapsed = time.perf_counter() - t0

        r = page_results[0]
        output = r.output.image

        # Metrics
        delta = np.abs(output.astype(float) - clean_rgb.astype(float))
        wm_region = alpha_gt > 0.1
        non_wm = ~wm_region

        fidelity_mean = float(delta[non_wm].mean()) if non_wm.any() else 0.0
        fidelity_p99 = float(np.percentile(delta[non_wm], 99)) if non_wm.any() else 0.0
        residual = float(delta[wm_region].mean()) if wm_region.any() else 0.0

        # Save output
        out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{style}_result.png"), out_bgr)

        rec = {
            "style": style,
            "elapsed_s": round(elapsed, 2),
            "fidelity_mean": round(fidelity_mean, 3),
            "fidelity_p99": round(fidelity_p99, 3),
            "residual_in_wm": round(residual, 3),
            "repairs": [str(x) for x in r.verdict.repairs],
            "mask_coverage": round(r.mask.coverage(), 5),
            "passed": r.verdict.passed,
        }
        results.append(rec)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="wm.e2e_test")
    parser.add_argument("--out", default="out/e2e", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    samples_dir = out / "samples"
    manifest = generate(samples_dir)
    print(f"Generated {len(manifest)} test samples -> {samples_dir}")

    results = evaluate(manifest, out)
    print(f"\n{'Style':<20} {'Time':>6} {'Fid.mean':>9} {'Fid.p99':>8} {'Residual':>9} {'Repairs':>20}")
    print("-" * 80)
    for r in results:
        print(f"{r['style']:<20} {r['elapsed_s']:>5.1f}s {r['fidelity_mean']:>9.3f} "
              f"{r['fidelity_p99']:>8.3f} {r['residual_in_wm']:>9.3f} "
              f"{','.join(r['repairs']):>20}")

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
