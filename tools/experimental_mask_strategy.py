"""Explicit research abstention for generative repair on data-bearing pages.

Binary detection alpha is not physical opacity. Source OCR cannot locate text
that it failed to read beneath a watermark. This candidate retains the entire
detection and declines generative repainting on an unverified data page; it
does not claim to have identified a physically unrecoverable pixel mask.
"""
from contextlib import contextmanager
from unittest.mock import patch

from wm.contract import RepairName


@contextmanager
def enabled(observations, *, capture=None):
    from wm import ocr, repair
    original = repair.repair

    def guarded(image, mask, thr):
        if mask.repair_mode != 'inpaint' or mask.method != 'recall_pixel_strokes':
            return original(image, mask, thr)
        raw = ocr.read_lines(image)
        body = []
        for box, text, confidence in raw or []:
            _, _, tilt = ocr._geometry(box)
            if confidence >= .45 and text.strip() and tilt <= 8:
                body.append(dict(box=box.tolist(), text=text, confidence=float(confidence)))
        # Three observed body items are a conservative page-level risk signal.
        # Their rectangles are never copied to detection or repair permission.
        abstain = raw is None or len(body) >= 3
        region = mask.footprint if mask.footprint is not None else mask.binary(thr.mask_alpha_min) > 0
        note = dict(version='data_page_abstention_v1', abstained=abstain,
                    source_ocr_available=raw is not None, source_body=body,
                    selected_pixels=int(region.sum()), physical_opacity_known=False,
                    unrecoverable_pixels=None, residual_quality=None, recognition_quality=None,
                    status='not_automatically_repaired' if abstain else 'needs_review',
                    reason=('原图有正文或OCR不可用，当前选区缺少可靠物理反解模型，暂停生成式重绘'
                            if abstain else '未触发本轮正文页面暂停条件；仍不能保证遮挡内容恢复正确'),
                    reference_read=False, experimental=True)
        observations.append(note)
        if capture:
            capture(region.copy() if abstain else region & False, note)
        if abstain:
            return image.copy(), [RepairName.NONE]
        return original(image, mask, thr)

    with patch.object(repair, 'repair', guarded):
        yield
