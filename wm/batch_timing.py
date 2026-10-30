"""Active time for one whole-batch run, independent of image quality metrics."""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter


@dataclass(frozen=True)
class TimingSnapshot:
    elapsed_seconds: float
    average_seconds: float | None
    completed: int
    total: int
    state: str
    failed: int = 0


class BatchTiming:
    def __init__(self, total: int):
        self.total = total
        self._lock = RLock()
        self._elapsed = 0.0
        self._started = perf_counter()
        self._completed = 0
        self._failed = 0
        self._completed_elapsed = 0.0
        self._state = "处理中"

    def _seconds(self) -> float:
        return self._elapsed + (perf_counter() - self._started
                                if self._started is not None else 0.0)

    def pause(self) -> None:
        with self._lock:
            if self._started is not None:
                self._elapsed = self._seconds()
                self._started = None
                self._state = "已暂停"

    def resume(self) -> None:
        with self._lock:
            if self._state == "已暂停":
                self._started = perf_counter()
                self._state = "处理中"

    def complete_page(self, *, failed: bool = False) -> None:
        with self._lock:
            self._completed += 1
            self._failed += int(failed)
            self._completed_elapsed = self._seconds()

    def finish(self, stopped: bool) -> None:
        with self._lock:
            self._elapsed = self._seconds()
            self._started = None
            self._state = "已停止" if stopped else "已完成"
            if self._completed == self.total:
                self._completed_elapsed = self._elapsed

    def snapshot(self) -> TimingSnapshot:
        with self._lock:
            return TimingSnapshot(
                self._seconds(),
                self._completed_elapsed / self._completed if self._completed else None,
                self._completed, self.total, self._state, self._failed,
            )


def format_duration(seconds: float) -> str:
    """Tenths of a second, with minutes/hours only when needed."""
    tenths = max(0, round(seconds * 10))
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    value = f"{rest / 10:.1f} 秒"
    if minutes or hours:
        value = f"{minutes} 分 {value}"
    if hours:
        value = f"{hours} 小时 {value}"
    return value
