"""A/B the OCR keep-out channel: same inputs, detector with and without it.

    .venv/bin/python tools/ocr_ab.py [--out out/ab]

Two arms, one variable. The synthetic set carries ground-truth alpha so mask
precision / recall are measured directly; the real samples have no ground truth,
so they are scored on a region derived from the *input* only - and the same
region for both arms, because the arm that draws a smaller mask would otherwise
win by shrinking its own denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm import e2e_test, masks, ocr, pipeline, repair  # noqa: E402
from wm.contract import EngineKind, Thresholds  # noqa: E402

REAL_SAMPLES = sorted(Path.home().glob("Pictures/水印/*.jpg"))


def _gray_dev(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray = img.astype(np.float32).mean(2)
    return gray, float(np.percentile(gray, 97))


def roi_of(src: np.ndarray, keepout: np.ndarray) -> np.ndarray:
    """Where a watermark could be, judged from the input alone.

    Everything the OCR says is body type is excluded, so this answers "how much
    of the non-text ink did we clear" and nothing else.
    """
    gray, P = _gray_dev(src)
    band = (P - gray) > 10
    h, w = gray.shape
    window = np.zeros(gray.shape, bool)
    window[int(h * .15):int(h * .85), int(w * .15):int(w * .85)] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    return cv2.dilate((band & ~keepout & window).astype(np.uint8), kernel).astype(bool)


def score_real(src: np.ndarray, out: np.ndarray, roi: np.ndarray,
               keepout: np.ndarray) -> dict:
    gs, P = _gray_dev(src)
    go, _ = _gray_dev(out)
    dev_s, dev_o = (P - gs)[roi], (P - go)[roi]
    changed = np.abs(out.astype(np.int16) - src.astype(np.int16)).mean(2)
    inside = keepout[changed > 4]
    return {
        "roi_fraction": round(float(roi.mean()), 5),
        "removal_ratio": round(float(1.0 - dev_o.mean() / max(dev_s.mean(), 1e-3)), 4),
        "roi_dev_after": round(float(dev_o.mean()), 2),
        "text_damage": round(float(inside.sum() / max(keepout.sum(), 1)), 5),
    }


def run(paths: list[dict], mode: str, out_dir: Path) -> dict[str, dict]:
    """mode: "classic" (no keep-out at all), "widened" (relaxed thresholds with an
    empty keep-out, i.e. the risk taken unprotected), "ocr" (relaxed + real zones)."""
    thr = Thresholds()
    rows: dict[str, dict] = {}
    for item in paths:
        name, img, alpha_gt, keepout_ref = (item["name"], item["image"],
                                            item["alpha"], item["roi_keepout"])
        t0 = time.perf_counter()
        if mode == "classic":
            keepout = None
        elif mode == "widened":
            keepout = np.zeros(img.shape[:2], bool)
        else:
            keepout = ocr.text_zones(img)
        ocr_ms = int((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        mask = masks.estimate_single_page(img, thr, keepout=keepout, use_ocr=(mode == "ocr"))
        mask_ms = int((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        out, repairs = repair.repair(img, mask, thr)
        repair_ms = int((time.perf_counter() - t0) * 1000)

        row = {"coverage": round(mask.coverage(thr.mask_alpha_min), 5),
               "footprint": round(float(mask.footprint.mean())
                                  if mask.footprint is not None else 0.0, 5),
               "confidence": round(mask.confidence, 4),
               "ocr_ms": ocr_ms, "mask_ms": mask_ms, "repair_ms": repair_ms,
               "total_s": round((ocr_ms + mask_ms + repair_ms) / 1000, 1),
               "veil_s": [round(float(x), 3) for x in mask.veil_strength],
               "veil_t": [round(float(x), 1) for x in mask.veil_offset],
               "repairs": [str(r) for r in repairs]}
        if alpha_gt is not None:
            report = pipeline.verdict_mod.build_verdict(
                source=img, output=out, mask=mask, engine=EngineKind.RASTER,
                repairs=repairs, elapsed_ms=ocr_ms + mask_ms + repair_ms, stage_ms={},
                thr=thr, alpha_reference=alpha_gt)
            row.update({k: round(report.metrics[k], 4)
                        for k in ("mask_precision", "mask_recall", "mask_iou")})
            row["passed"] = report.passed
            row["failures"] = sorted(str(f) for f in report.failures)
        else:
            row.update(score_real(img, out, roi_of(img, keepout_ref), keepout_ref))

        stem = f"{mode}_{name}"
        cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if mask.footprint is not None:
            vis = out.astype(np.float32)
            vis[mask.footprint] = (vis[mask.footprint] * 0.55
                                   + np.array([255, 40, 40]) * 0.45)
            cv2.imwrite(str(out_dir / f"{stem}_footprint.png"),
                        cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
        rows[name] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="ocr_ab")
    ap.add_argument("--out", default="out/ab")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[dict] = []
    for p in REAL_SAMPLES:
        img = np.asarray(Image.open(p).convert("RGB"))
        # One shared keep-out per input, so both arms are scored on one ROI.
        paths.append({"name": p.stem, "image": img, "alpha": None,
                      "roi_keepout": ocr.text_zones(img)})
    for entry in e2e_test.generate(out_dir / "syn"):
        paths.append({
            "name": "syn_" + entry["style"],
            "image": cv2.cvtColor(cv2.imread(entry["watermarked"]), cv2.COLOR_BGR2RGB),
            "alpha": cv2.imread(entry["alpha"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0,
            "roi_keepout": np.zeros((1, 1), bool)})

    arms = {mode: run(paths, mode, out_dir) for mode in ("classic", "widened", "ocr")}

    def cell(rows: dict, name: str, key: str, fmt: str = "{:.4f}") -> str:
        v = rows[name].get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "  --  "

    print(f"\n{'sample':<26} {'coverage  classic/widened/ocr':>34}  detail (ocr arm)")
    print("-" * 112)
    for name in arms["classic"]:
        cov = "  ".join(cell(arms[m], name, "coverage") for m in arms)
        if "removal_ratio" in arms["classic"][name]:
            detail = ("removal " + "/".join(cell(arms[m], name, "removal_ratio", "{:.3f}") for m in arms)
                      + "  text_damage " + "/".join(
                          cell(arms[m], name, "text_damage", "{:.4f}") for m in arms))
        else:
            detail = ("P " + "/".join(cell(arms[m], name, "mask_precision", "{:.3f}") for m in arms)
                      + "  R " + "/".join(cell(arms[m], name, "mask_recall", "{:.3f}") for m in arms)
                      + "  pass " + "/".join(str(arms[m][name].get("passed"))[:4] for m in arms))
        print(f"{name:<26} {cov:>34}  {detail}")
    print("\nseconds/page  " + "  ".join(
        f"{m}: max {max(r['total_s'] for r in arms[m].values()):.0f}s" for m in arms))
    print("OCR cost      " + "  ".join(
        f"{m}: max {max(r['ocr_ms'] for r in arms[m].values())}ms" for m in arms))
    (out_dir / "ab_results.json").write_text(
        json.dumps(arms, indent=2, ensure_ascii=False), "utf-8")
    print(f"-> {out_dir / 'ab_results.json'}")


if __name__ == "__main__":
    main()
