"""诊断 _document_overlay 拟合在哪一步拒绝（开发用，不入库）。"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm import masks, ocr, repeated  # noqa: E402
from wm.contract import Thresholds  # noqa: E402


def trace(img, region, profile, confidence):
    keepout = ocr.text_zones(img)
    print("  keepout:", None if keepout is None else int(keepout.sum()))
    if keepout is None or not keepout.any():
        print("  REJECT: no keepout")
        return
    gray = img.mean(2)
    paper = float(np.percentile(gray, 97))
    edges = np.arange(0, 258, 2)
    histogram, _ = np.histogram(gray[region & ~keepout], bins=edges)
    histogram[(edges[:-1] < .16 * paper) | (edges[:-1] > .78 * paper)] = 0
    print(f"  paper={paper:.1f} hist_max={histogram.max()} (需>=20)")
    if histogram.max() < 20:
        print("  REJECT: covered-paper pole too weak")
        return
    covered = float(edges[np.argmax(histogram)] + 1)
    estimated = np.clip((paper - profile) / max(paper - covered, 1), 0, 1)
    ink_pixels = gray[region & keepout & (estimated > .7) &
                      (gray > .05 * paper) & (gray < .65 * covered)]
    print(f"  covered={covered:.1f} ink_pixels={ink_pixels.size} (需>=20)")
    if ink_pixels.size < 20:
        print("  REJECT: covered-ink pole missing")
        return
    ink = float(np.percentile(ink_pixels, 10))
    transmittance = (covered - ink) / paper
    print(f"  ink={ink:.1f} transmittance={transmittance:.3f} (需 .08-.85)")


def main():
    sid = sys.argv[1]
    real = Path("../real_samples")
    import json
    m = json.loads((real / "manifest.json").read_text("utf-8"))
    row = next(s for s in m["samples"] if s["id"] == sid)
    img = np.asarray(Image.open(real / row["jpg"]).convert("RGB"))
    print(f"== {sid} ==")
    orig = repeated._document_overlay
    repeated._document_overlay = lambda im, region, profile, conf: (
        trace(im, region, profile, conf), orig(im, region, profile, conf))[1]
    mask = masks.estimate_single_page(img, Thresholds())
    print("  method:", mask.method, "coverage:", round(mask.coverage(), 4))


if __name__ == "__main__":
    main()
