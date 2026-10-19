from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import fixtures, pipeline, synth
from .contract import PageImage, Thresholds
from .pipeline import GroundTruth


def load_doc(doc_dir: Path, dpi: int) -> tuple[list[PageImage], GroundTruth]:
    doc_id = doc_dir.name
    pages: list[PageImage] = []
    gt: GroundTruth = {}
    for npz_path in sorted(doc_dir.glob("page*_gt.npz")):
        index = int(npz_path.name[4:6])
        record = np.load(npz_path)
        watermarked = np.asarray(Image.open(
            npz_path.with_name(npz_path.name.replace("_gt.npz", ".png"))).convert("RGB"))
        strength = float(record["strength"])
        gt[f"{doc_id}#{index}"] = (
            record["clean"], record["alpha"],
            (np.full(3, 1.0 - strength, np.float32),
             (record["color"] * strength).astype(np.float32)),
        )
        pages.append(PageImage(image=watermarked, dpi=dpi, doc_id=doc_id, page_index=index))
    return pages, gt


def _agg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    return statistics.fmean(vals) if vals else float("nan")


def cmd_synth(args: argparse.Namespace) -> None:
    root = synth.generate_dataset(args.out, docs=args.docs, pages_per_doc=args.pages,
                                 dpi=args.dpi, jpeg_q=args.jpeg, noise=args.noise)
    styles = sorted({p["doc_id"].split("_", 1)[1] for p in json.loads(
        (root / "dataset.json").read_text("utf-8"))["pages"]})
    print(f"dataset -> {root}  docs={args.docs} pages={args.pages} dpi={args.dpi} "
          f"styles={len(styles)}")


def cmd_eval(args: argparse.Namespace) -> None:
    root = Path(args.data)
    thr = Thresholds()
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pages_passed = {"single": 0, "cross": 0}
    pages_total = {"single": 0, "cross": 0}
    failures: dict[str, dict[str, int]] = {"single": defaultdict(int), "cross": defaultdict(int)}

    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        style = doc_dir.name.split("_", 1)[1]
        pages, gt = load_doc(doc_dir, args.dpi)
        for mode, force_single in (("single", True), ("cross", False)):
            results = pipeline.process_pages(pages, thr, gt, force_single=force_single)
            for r in results:
                buckets[(style, mode)].append(
                    r.verdict.metrics | {"passed": float(r.verdict.passed),
                                         "elapsed": float(r.verdict.elapsed_ms)})
                pages_total[mode] += 1
                if r.verdict.passed:
                    pages_passed[mode] += 1
                for f in r.verdict.failures:
                    failures[mode][str(f)] += 1

    head = f"{'style':<12} {'mask':<7} {'IoU':>6} {'prec':>6} {'rec':>6} {'s_err':>7} " \
           f"{'PSNR':>7} {'SSIM':>6} {'PSNRm':>7} {'inkIoU':>7} {'dmg':>6} {'res':>6} {'fidΔ':>6} {'resid':>6} {'pass%':>6} {'ms':>6}"
    print(head)
    print("-" * len(head))
    for (style, mode), rows in sorted(buckets.items()):
        print(f"{style:<12} {mode:<7} {_agg(rows,'mask_iou'):>6.3f} {_agg(rows,'mask_precision'):>6.3f} "
              f"{_agg(rows,'mask_recall'):>6.3f} {_agg(rows,'veil_s_error'):>7.3f} "
              f"{_agg(rows,'psnr'):>7.2f} {_agg(rows,'ssim'):>6.3f} {_agg(rows,'psnr_in_mask'):>7.2f} "
              f"{_agg(rows,'ink_iou'):>7.3f} {_agg(rows,'ink_damage'):>6.3f} "
              f"{_agg(rows,'ink_residue'):>6.3f} {_agg(rows,'fidelity_mean_delta'):>6.2f} "
              f"{_agg(rows,'residual_coverage'):>6.3f} {100*_agg(rows,'passed'):>6.1f} "
              f"{_agg(rows,'elapsed'):>6.0f}")

    print("\nsuccess rate (four-condition protocol):")
    for mode in ("single", "cross"):
        total = pages_total[mode] or 1
        print(f"  {mode:<7} {pages_passed[mode]}/{total} = {100*pages_passed[mode]/total:.1f}%"
              f"   binding failures: {dict(failures[mode])}")


def cmd_fixtures(args: argparse.Namespace) -> None:
    for doc_dir in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        info = fixtures.build(doc_dir, Path(args.out) / doc_dir.name)
        print(f"{doc_dir.name}: {info['pages']} pages -> {info['out']}")


def cmd_run(args: argparse.Namespace) -> None:
    results = pipeline.process_file(args.input, dpi=args.dpi, skip_structure=args.force_raster)
    pipeline.write_results(results, args.out)
    for r in results:
        print(json.dumps(r.to_dict(), ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wm", description="离线智能去水印原型")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="生成带真值的合成样本集")
    p.add_argument("--out", default="data/syn")
    p.add_argument("--docs", type=int, default=5)
    p.add_argument("--pages", type=int, default=6)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--jpeg", type=int, default=0)
    p.add_argument("--noise", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("eval", help="在合成集上按四条件协议统计成功率")
    p.add_argument("--data", default="data/syn")
    p.add_argument("--dpi", type=int, default=200)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("fixtures", help="把合成集打成光栅 PDF 与结构水印 PDF")
    p.add_argument("--data", default="data/syn")
    p.add_argument("--out", default="data/pdf")
    p.set_defaults(fn=cmd_fixtures)

    p = sub.add_parser("run", help="处理一个 PDF 或图片")
    p.add_argument("input")
    p.add_argument("--out", default="out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--force-raster", action="store_true")
    p.set_defaults(fn=cmd_run)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
