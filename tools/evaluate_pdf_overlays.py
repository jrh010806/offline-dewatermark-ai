"""Evaluate source-only PDF operator removal; references are scoring-only.

Requires an audited paired inventory (tools/evaluate_samples.py inventory).
Writes rewritten vector PDFs under the chosen output folder, never beside inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.pdf_text import strip_translucent_text


def raster(page, dpi=100):
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path('out/new_samples/inventory.json'))
    parser.add_argument('--out', type=Path, default=Path('out/new_samples/pdf_outputs'))
    args = parser.parse_args()
    inv = json.loads(args.inventory.read_text()); base = Path(inv['root'])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in inv['rows']:
        if item['kind'] != 'pdf':
            continue
        path = base/item['watermarked']; source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == item['watermarked_sha256']
        start = time.perf_counter()
        output, counts = strip_translucent_text(source)
        ms = round((time.perf_counter()-start)*1000)
        if any(counts):
            (args.out/f"{item['id']}_clean.pdf").write_bytes(output)
        # Only now read the clean reference; it never enters detection/removal.
        clean = (base/item['clean']).read_bytes()
        negative, negative_counts = strip_translucent_text(clean)
        row = dict(id=item['id'], type=item['watermark_type'], split=item['split'],
                   text_blocks_removed=counts, strip_ms=ms,
                   clean_negative_untouched=negative == clean,
                   clean_negative_counts=negative_counts, pages=[])
        with fitz.open(stream=output, filetype='pdf') as got, \
             fitz.open(stream=clean, filetype='pdf') as ref, \
             fitz.open(stream=source, filetype='pdf') as src:
            for n in range(len(src)):
                a, b, c = raster(src[n]), raster(got[n]), raster(ref[n])
                row['pages'].append(dict(index=n,
                    input_mae=float(abs(a.astype(float)-c).mean()),
                    output_mae=float(abs(b.astype(float)-c).mean()),
                    exact_reference=bool(np.array_equal(b, c)),
                    original_exact=bool(np.array_equal(a, b))))
                if n == 0:
                    Image.fromarray(np.concatenate([a, b, c], axis=1)).save(args.out/f"{item['id']}_comparison.png")
        rows.append(row)
        (args.out/'metrics.json').write_text(json.dumps(rows, indent=2))
        print(item['id'], item['watermark_type'], 'blocks', sum(counts),
              'exact', sum(p['exact_reference'] for p in row['pages']), '/', len(row['pages']),
              'negative unchanged', row['clean_negative_untouched'], flush=True)


if __name__ == '__main__':
    main()
