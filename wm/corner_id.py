"""Corner identifier watermarks: small ID-like text in page corners.

Detection is evidence-gated like every other channel: a mostly white page,
an OCR box inside a corner crop, and ID-shaped content (letters+digits or
dashes, not a page number or body sentence). Blank-paper areas are restored
by paper fill; photo backgrounds fall back to a bounded inpaint.
"""
from __future__ import annotations

import re
import cv2
import numpy as np

from . import ocr
from .contract import MaskLayer, MaskSource

_ID = re.compile(r"^(?=.*[A-Za-z0-9].*)[A-Za-z0-9][A-Za-z0-9\-_/\.]{3,}$")
_WORD = re.compile(r"[A-Za-z]")


def _id_like(text: str) -> bool:
    t = text.strip().replace(" ", "")
    if len(t) < 5 or not _ID.match(t):
        return False
    # Page numbers and pure words are not identifiers.
    has_digit = any(c.isdigit() for c in t)
    return has_digit and bool(_WORD.search(t))


def estimate(img: np.ndarray) -> MaskLayer | None:
    gray = img.mean(2).astype(np.float32)
    h, w = gray.shape
    paper = float(np.percentile(gray, 97))
    white_page = paper >= 180 and (gray > 235).mean() >= .45
    cw, ch = int(w * .45), int(h * .20)
    corners = [(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)]
    regions = []
    for x0, y0 in corners:
        crop = img[y0:y0 + ch, x0:x0 + cw]
        if crop.size == 0:
            continue
        up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for box, text, confidence in ocr.read_lines(up) or []:
            if confidence < .45 or not _id_like(text):
                continue
            box = box / 2.0
            height, length, tilt = ocr._geometry(box)
            if height > 90 or tilt > 25:
                continue
            bx0 = int(max(0, box[:, 0].min() - 3))
            bx1 = int(min(cw, box[:, 0].max() + 3))
            by0 = int(max(0, box[:, 1].min() - 3))
            by1 = int(min(ch, box[:, 1].max() + 3))
            regions.append((x0 + bx0, y0 + by0, x0 + bx1, y0 + by1))
    if not regions:
        return None
    alpha = np.zeros(gray.shape, np.float32)
    for x0, y0, x1, y1 in regions:
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        text_pixels = patch < paper - 18
        if text_pixels.mean() < .01 or text_pixels.mean() > .8:
            continue
        alpha[y0:y1, x0:x1][text_pixels] = 1.0
    if not .00001 < alpha.mean() < .05:
        return None
    alpha = cv2.dilate(alpha, np.ones((2, 2), np.uint8))
    color = np.median(img[alpha > 0], axis=0).astype(np.float32)
    paper_img = np.full(img.shape, paper, np.float32)
    return MaskLayer(alpha=alpha, source=MaskSource.PRIOR, confidence=.8,
                     veil_strength=np.zeros(3, np.float32), veil_offset=color,
                     paper=paper_img, footprint=alpha > 0,
                     repair_mode='corner_cleanup' if white_page else 'inpaint',
                     method='corner_id_text', text_keepout=None)
