"""OCR keep-out zones: where the detector is forbidden to claim watermark.

Tonal detection is blind in exactly one direction - a faint watermark stroke and
an antialiased glyph edge sit in the same mid-grey band - so widening its
threshold to catch more watermark also erases text, which is the clause the
tender penalises hardest. A text detector breaks the tie from the other side: it
knows where glyphs *are* by reading them, not by measuring them, so the
threshold can be relaxed everywhere else.

The catch is that a stamp watermark is also text, so the boxes cannot be
adopted wholesale. Only boxes matching the page's own body type are kept out:
comparable glyph height, essentially horizontal, read with confidence. A large
rotated low-contrast cluster - a seal, a diagonal company name - fails that
profile and is left for the watermark detector to claim.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any

import cv2
import numpy as np

_MAX_CACHE = 8

_lock = threading.Lock()
_engine: Any = None
_engine_failed = False
_cache: dict[str, np.ndarray] = {}
_line_cache: dict[str, list] = {}


def read_lines(img: np.ndarray) -> list | None:
    """OCR observations shared by body protection and watermark localization.

    Return None when OCR is unavailable, rather than claiming an empty page.
    Confidence is normalized because RapidOCR versions also return strings.
    """
    key = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
    if key in _line_cache:
        return _line_cache[key]
    engine = _get_engine()
    if engine is None:
        return None
    result, _ = engine(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)
    lines = []
    for item in result or []:
        try:
            box, text, confidence = np.asarray(item[0], np.float32), str(item[1]), float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if box.shape == (4, 2) and np.isfinite(box).all() and np.isfinite(confidence):
            lines.append((box, text, confidence))
    if len(_line_cache) >= _MAX_CACHE:
        _line_cache.pop(next(iter(_line_cache)))
    _line_cache[key] = lines
    return lines


def _get_engine():
    """None on failure, and the caller must degrade to no-keep-out silently.

    RapidOCR is an optional dependency: the classic channel ships on machines
    without it, and a missing OCR model must not take detection down with it.
    """
    global _engine, _engine_failed
    with _lock:
        if _engine is not None or _engine_failed:
            return _engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except Exception:
            _engine_failed = True
        return _engine


def available() -> bool:
    """False when the optional OCR dependency is missing; detection then runs
    its conservative no-keep-out path."""
    return _get_engine() is not None


def _boxes(result) -> list[np.ndarray]:
    out = []
    for item in result or []:
        try:
            box, conf = np.asarray(item[0], np.float32), float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if conf >= 0.5 and box.shape == (4, 2):
            out.append(box)
    return out


def _geometry(box: np.ndarray) -> tuple[float, float, float]:
    """(glyph height, line length, tilt in degrees).

    PP-OCR orders a box tl, tr, br, bl, so edge 0-1 is the reading direction and
    edge 1-2 is the glyph height; taking both edges and sorting keeps this right
    even when a page comes in rotated.
    """
    a = float(np.linalg.norm(box[1] - box[0]))
    b = float(np.linalg.norm(box[2] - box[1]))
    height, length = min(a, b), max(a, b)
    tilt = abs(np.degrees(np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0]))) % 90
    return height, length, min(tilt, 90.0 - tilt)


def body_text_boxes(img: np.ndarray) -> list[np.ndarray] | None:
    """Boxes that are body type, or None when OCR is unavailable.

    ``height_tol`` is deliberately loose on the small side: a caption or a table
    cell in 8 pt is far shorter than the 14 pt body and is still ink that must
    survive. It is tight on the large side and on tilt, which is where a seal or
    a diagonal overlay sits and where the watermark detector needs the room.
    """
    result = read_lines(img)
    if result is None:
        return None
    boxes = _boxes(result)
    if not boxes:
        return []
    geos = [_geometry(b) for b in boxes]
    modal = float(np.median([g[0] for g in geos]))
    return [box for box, (h, _length, tilt) in zip(boxes, geos)
            if h <= 2.5 * modal and tilt <= 15.0]


def text_zones(img: np.ndarray, pad_ratio: float = 0.6) -> np.ndarray | None:
    """Boolean keep-out: body glyphs plus the antialiasing ramp around them.

    The pad is proportional to the glyph height because that ramp is; a fixed
    pixel pad would either miss it on 9 pt type or smother the watermark on 24 pt.
    """
    key = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
    hit = _cache.get(key)
    if hit is not None:
        return hit
    boxes = body_text_boxes(img)
    if boxes is None:
        return None
    zones = np.zeros(img.shape[:2], np.uint8)
    for box in boxes:
        cv2.fillPoly(zones, [box.astype(np.int32)], 1)
    if boxes:
        pad = max(2, int(pad_ratio * float(np.median([_geometry(b)[0] for b in boxes]))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        zones = cv2.dilate(zones, kernel)
    out = zones.astype(bool)
    if len(_cache) >= _MAX_CACHE:
        _cache.pop(next(iter(_cache)))
    _cache[key] = out
    return out
