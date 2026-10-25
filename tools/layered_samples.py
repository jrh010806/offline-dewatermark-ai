"""分层评测样本生成（迭代零，Q8）。

以 real_samples 的干净底图为基础，按 样式 × 透明度 × 底材 分层生成带真值样本，
输出与 real_samples 完全相同的目录格式（clean/jpg/mask/manifest.json），
可直接用 tools/run_full_audit.py real --real-root <输出目录> 评测。

    .venv/bin/python tools/layered_samples.py --real-root ../real_samples --out ../layered_samples
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
from wm.synth import apply_watermark, make_watermark  # noqa: E402

STYLES = ("tiled_text", "big_center", "dot_grid", "heavy_dark", "corner_id")
FACTORS = (("light", 0.55), ("medium", 1.0), ("dark", 1.5))
REPEATED = {"tiled_text", "dot_grid", "heavy_dark"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-root", type=Path, default=Path("../real_samples"))
    ap.add_argument("--out", type=Path, default=Path("../layered_samples"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--jpeg-q", type=int, default=92)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--docs", type=int, default=2, help="文档底图数量")
    ap.add_argument("--photos", type=int, default=2, help="照片底图数量")
    args = ap.parse_args()

    real = json.loads((args.real_root / "manifest.json").read_text("utf-8"))
    bases: dict[str, dict] = {}
    for s in real["samples"]:
        if s["base"] not in bases:
            bases[s["base"]] = dict(base=s["base"], base_kind=s["base_kind"],
                                    clean=args.real_root / s["clean"])
    docs = sorted((b for b in bases.values() if b["base_kind"] == "pdf_page"), key=lambda b: b["base"])[: args.docs]
    photos = sorted((b for b in bases.values() if b["base_kind"] == "photo"), key=lambda b: b["base"])[: args.photos]
    chosen = docs + photos
    if not chosen:
        raise SystemExit("real_samples 中没有可用底图")

    out = args.out
    for sub in ("clean", "jpg", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "dpi": args.dpi, "jpeg_q": args.jpeg_q,
                "note": "分层能力评测集：真底材+合成水印，含真值与可解性标注",
                "samples": []}
    rng = random.Random(args.seed)
    n = 0
    for style in STYLES:
        for factor_name, factor in FACTORS:
            for b in chosen:
                clean = np.asarray(Image.open(b["clean"]).convert("RGB"))
                alpha, color, base_strength = make_watermark(clean.shape, style, args.dpi, n, rng)
                strength = float(np.clip(base_strength * factor, 0.05, 0.95))
                wm_img = apply_watermark(clean, alpha, color, strength)
                sid = f"{style}__{factor_name}__{b['base_kind']}_{b['base']}"
                Image.fromarray(clean).save(out / "clean" / f"{sid}.png")
                buf = io.BytesIO()
                Image.fromarray(wm_img).save(buf, "JPEG", quality=args.jpeg_q)
                Image.open(buf).convert("RGB").save(out / "jpg" / f"{sid}.jpg")
                Image.fromarray((alpha * 255).astype(np.uint8)).save(out / "mask" / f"{sid}.png")
                opaque = bool((alpha * strength >= 0.97).any())
                manifest["samples"].append({
                    "id": sid, "style": style, "base": b["base"], "base_kind": b["base_kind"],
                    "jpg": f"jpg/{sid}.jpg", "clean": f"clean/{sid}.png", "mask": f"mask/{sid}.png",
                    "strength": round(strength, 3),
                    "color": [round(float(c), 1) for c in color],
                    "strength_layer": factor_name,
                    "repeated": style in REPEATED,
                    "solvability": "含全遮挡" if opaque else "可解",
                })
                n += 1
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"共 {n} 张分层样本（{len(STYLES)} 样式 × {len(FACTORS)} 透明度 × {len(chosen)} 底图）-> {out}")


if __name__ == "__main__":
    main()
