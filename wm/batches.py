"""Session batches and a cooperative, page-by-page desktop queue."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition
from time import perf_counter
from uuid import uuid4
import logging

from PySide6.QtCore import QThread, Signal

from . import masks, pipeline, structure
from .contract import MaskLayer, PageImage, PageResult, Thresholds


@dataclass
class Batch:
    name: str
    pages: list[PageImage]
    uid: str = field(default_factory=lambda: uuid4().hex)
    params: dict[str, int] = field(default_factory=dict)
    use_ocr: bool = True
    results: dict[int, PageResult] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    states: dict[int, str] = field(default_factory=dict)
    priors: dict[tuple, MaskLayer] = field(default_factory=dict)
    dirty: set[int] = field(default_factory=set)

    def invalidate(self) -> None:
        self.priors.clear()


class BatchWorker(QThread):
    update = Signal(object)

    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self.jobs = jobs  # (batch, selected indices, frozen thresholds, close_px, OCR)
        self.total = sum(len(indices) for _, indices, *_ in jobs)
        self.done = 0
        self.outcomes = []
        self.stopped = False
        self._paused = False
        self._condition = Condition()

    def pause(self, paused: bool) -> None:
        with self._condition:
            self._paused = paused
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self.stopped = True
            self._condition.notify_all()

    def _checkpoint(self) -> bool:
        with self._condition:
            while self._paused and not self.stopped:
                self.update.emit(dict(kind="paused", done=self.done, total=self.total))
                self._condition.wait()
            return not self.stopped

    def run(self) -> None:
        for batch, indices, thr, close_px, use_ocr in self.jobs:
            groups = {}
            unavailable_priors = set()
            for page in batch.pages:
                groups.setdefault(page.image.shape, []).append(page)
            for index in indices:
                if not self._checkpoint():
                    return
                page = batch.pages[index]
                group = groups[page.image.shape]
                key = (page.image.shape, thr.mask_alpha_min, close_px, use_ocr)
                try:
                    started = perf_counter()
                    self.update.emit(dict(kind="stage", batch=batch.uid, index=index,
                                          message="检查 PDF 透明文字对象"))
                    result = structure.process_text_page(page, thr)
                    if result is None:
                        self.update.emit(dict(kind="stage", batch=batch.uid, index=index,
                                              message="统计同尺寸组跨页先验" if len(group) > 1 else "单张组：单图识别"))
                        if len(group) > 1:
                            mask = batch.priors.get(key)
                            if mask is None and key not in unavailable_priors:
                                try:
                                    mask = masks.estimate_crosspage([p.image for p in group], thr,
                                                                    close_px=close_px)
                                    batch.priors[key] = mask
                                except Exception as exc:
                                    logging.getLogger(__name__).warning(
                                        "Cross-page prior unavailable; using single-image detection: %s", exc)
                                    unavailable_priors.add(key)
                            single = masks.estimate_single_page(
                                page.image, thr, close_px=close_px,
                                use_ocr=use_ocr)
                            mask = masks.with_prior(single, mask, thr)
                        else:
                            mask = masks.estimate_single_page(page.image, thr, close_px=close_px,
                                                              use_ocr=use_ocr)
                        mask_ms = int((perf_counter() - started) * 1000)
                        # Pause/stop requests made during prior estimation take effect before repair.
                        if not self._checkpoint():
                            return
                        result = pipeline._process_one(page, mask, thr, mask_ms=mask_ms)
                    elif not self._checkpoint():
                        return
                    if isinstance(result, PageResult):
                        result.processing.update(mask_close_px=close_px, ocr_enabled=use_ocr)
                    payload = dict(result=result, error=None)
                except Exception as exc:
                    payload = dict(result=None, error=str(exc))
                result = payload["result"]
                self.outcomes.append("运行失败" if result is None else
                                     "质量未通过" if not result.verdict.passed else
                                     "待复核" if result.verdict.review_needed else "通过")
                self.done += 1
                self.update.emit(dict(kind="page", batch=batch.uid, index=index,
                                      done=self.done, total=self.total, **payload))
