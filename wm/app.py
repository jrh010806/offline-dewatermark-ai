"""Desktop shell. Talks to the pipeline only through the contract types, so the
algorithm track and this file can evolve independently.

    python -m wm.app [file]            interactive
    python -m wm.app --smoke [file]    offscreen self-check, no window shown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import documents, pipeline
from .contract import JobEvent, JobState, PageImage, PageResult, Thresholds

VIEW_MODES = ("结果", "原图", "掩码叠加", "分屏对比")
OPEN_FILTER = "可处理文档 (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"


def make_app() -> QApplication:
    """Qt's fallback font has no CJK glyphs, so every label renders as a box."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    font = QFont()
    font.setFamilies(["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "WenQuanYi Micro Hei"])
    font.setPointSize(9)
    app.setFont(font)
    return app


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3,
                  QImage.Format_RGB888).copy()


def overlay_mask(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    tint = np.array([255, 40, 40], np.float32)
    a = (alpha[..., None] * 0.55).astype(np.float32)
    return np.clip(img.astype(np.float32) * (1 - a) + tint * a, 0, 255).astype(np.uint8)


def split_compare(before: np.ndarray, after: np.ndarray, frac: float = 0.5) -> np.ndarray:
    out = after.copy()
    cut = int(after.shape[1] * frac)
    out[:, :cut] = before[:, :cut]
    line = np.zeros((after.shape[0], 4, 3), np.uint8)
    out[:, max(cut - 2, 0):cut + 2] = line
    return out


class Worker(QThread):
    event = Signal(dict)
    result_ready = Signal(object)
    finished_all = Signal()

    def __init__(self, pages: list[PageImage], parent=None):
        super().__init__(parent)
        self.pages = pages
        self.results: list[PageResult] = []

    def run(self) -> None:
        thr = Thresholds()
        total = len(self.pages)
        self.event.emit(JobEvent("local", "batch", JobState.RUNNING, 0, total).to_dict())
        passed = 0
        review = 0
        for i, result in enumerate(pipeline.process_pages(self.pages, thr)):
            self.results.append(result)
            state = (JobState.FAILED if not result.verdict.passed else
                     JobState.NEEDS_REVIEW if result.verdict.review_needed else JobState.PASSED)
            passed += state is JobState.PASSED
            review += state is JobState.NEEDS_REVIEW
            self.result_ready.emit(result)
            self.event.emit(JobEvent(
                "local", result.source.key, state,
                i + 1, total, elapsed_ms=result.verdict.elapsed_ms,
                passed_count=passed, review_count=review,
                failure_codes=[str(f) for f in result.verdict.failures],
            ).to_dict())
        self.finished_all.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("离线智能去水印 · 原型")
        self.resize(1500, 900)
        self.pages: list[PageImage] = []
        self.results: list[PageResult] = []
        self._current = -1

        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self.show_page)
        self.canvas = QLabel("打开 PDF 或图片开始")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.mode_box = QComboBox()
        self.mode_box.addItems(VIEW_MODES)
        self.mode_box.currentTextChanged.connect(self.render_canvas)
        self.open_btn = QPushButton("打开文档")
        self.open_btn.clicked.connect(self.open_file)
        self.run_btn = QPushButton("处理全部")
        self.run_btn.clicked.connect(self.start_run)
        self.run_btn.setEnabled(False)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["指标", "值"])
        self.conditions = QLabel("判定：—")
        self.conditions.setWordWrap(True)
        self.progress = QProgressBar()
        self.status = QLabel("就绪")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("页面"))
        left_layout.addWidget(self.page_list, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_btn)
        buttons.addWidget(self.run_btn)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("视图"))
        right_layout.addWidget(self.mode_box)
        right_layout.addWidget(QLabel("逐页指标"))
        right_layout.addWidget(self.metrics, 1)
        right_layout.addWidget(self.conditions)

        splitter = QSplitter()
        splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addLayout(buttons)
        center_layout.addWidget(self.canvas, 1)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([220, 900, 330])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.worker: Worker | None = None

    def open_file(self, path: str | Path | None = None) -> None:
        if not path:
            files, _ = QFileDialog.getOpenFileNames(
                self, "打开文档（可多选图片，同一水印的图片作为一批处理）",
                "data", OPEN_FILTER)
            if not files:
                return
            pages = documents.load_many(files)
        else:
            pages = documents.load_pages(path, dpi=200)
        self.pages = pages
        self.results = []
        self.page_list.clear()
        for page in self.pages:
            self.page_list.addItem(QListWidgetItem(f"{page.doc_id} · 第 {page.page_index + 1} 页"))
        self.run_btn.setEnabled(True)
        self.progress.setMaximum(len(self.pages))
        self.progress.setValue(0)
        self.show_page(0)

    def start_run(self) -> None:
        if not self.pages or self.worker and self.worker.isRunning():
            return
        self.results = []
        self.worker = Worker(self.pages, self)
        self.worker.event.connect(self.on_event)
        self.worker.result_ready.connect(self.results.append)
        self.worker.finished_all.connect(lambda: self.status.setText("完成"))
        self.worker.start()

    def on_event(self, payload: dict) -> None:
        self.progress.setValue(payload["index"])
        rate = payload.get("running_success_rate", 0.0)
        state = payload["state"]
        self.status.setText(
            f"{payload['page_key']} {state} · {payload['elapsed_ms']}ms · "
            f"四项全过 {rate * 100:.1f}% · 待复核 {payload.get('review_count', 0)}")
        row = payload["index"] - 1
        colour = {str(JobState.PASSED): Qt.darkGreen,
                  str(JobState.NEEDS_REVIEW): Qt.darkYellow,
                  str(JobState.FAILED): Qt.darkRed}.get(state)
        if row >= 0 and colour is not None:
            item = self.page_list.item(row)
            item.setBackground(colour)
            if row == self._current:
                self.show_page(row)

    def show_page(self, row: int) -> None:
        self._current = row
        if not (0 <= row < len(self.pages)):
            return
        if row < len(self.results):
            self._fill_metrics(self.results[row])
        self.render_canvas()

    def _fill_metrics(self, result: PageResult) -> None:
        report = result.verdict
        rows = [("引擎", str(report.engine)), ("掩码来源", str(result.mask.source)),
                ("耗时 ms", report.elapsed_ms), ("阶段", report.stage_ms)]
        rows += [(k, f"{v:.4f}" if isinstance(v, float) else v)
                 for k, v in sorted(report.metrics.items())]
        self.metrics.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(str(k)))
            self.metrics.setItem(i, 1, QTableWidgetItem(str(v)))
        flags = "  ".join(
            f"{'✔' if ok else '✘'}{name}" for name, ok in report.conditions.items())
        self.conditions.setText(f"判定：{flags}\n失败：{[str(f) for f in report.failures] or '无'}")

    def render_canvas(self) -> None:
        row = self._current
        if not (0 <= row < len(self.pages)):
            return
        source = self.pages[row].image
        result = self.results[row] if row < len(self.results) else None
        mode = self.mode_box.currentText()
        if result is None:
            img = source
        elif mode == "原图":
            img = source
        elif mode == "掩码叠加":
            img = overlay_mask(source, result.mask.alpha)
        elif mode == "分屏对比":
            img = split_compare(source, result.output.image)
        else:
            img = result.output.image
        pixmap = QPixmap.fromImage(to_qimage(img))
        self.canvas.setPixmap(pixmap.scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def smoke(path: str | None) -> int:
    app = make_app()
    win = MainWindow()
    win.show()
    win.open_file(path or "data/syn/doc00_tiled_text/page00.png")
    win.start_run()
    win.worker.wait(180_000)
    app.processEvents()
    assert win.results, "no result produced"
    report = win.results[0].verdict
    print(f"smoke ok: pages={len(win.pages)} engine={report.engine} "
          f"conditions={report.conditions} metrics={ {k: round(v, 4) for k, v in report.metrics.items()} } "
          f"elapsed={report.elapsed_ms}ms canvas={win.canvas.pixmap().width()}x{win.canvas.pixmap().height()}")
    return 0


def shot(path: str | None, out_dir: str) -> int:
    """Headless render of the workbench, one PNG per view mode, for demos and reports."""
    app = make_app()
    win = MainWindow()
    win.show()
    win.open_file(path or "data/pdf/doc00_tiled_text/raster_flattened.pdf")
    win.start_run()
    win.worker.wait(240_000)
    app.processEvents()
    win.page_list.setCurrentRow(0)
    app.processEvents()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(VIEW_MODES):
        win.mode_box.setCurrentText(mode)
        app.processEvents()
        win.grab().save(str(out / f"workbench{i}_{mode.split('(')[0]}.png"))
    print(f"shot -> {out} modes={len(VIEW_MODES)} results={len(win.results)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="wm.app")
    parser.add_argument("file", nargs="?", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shot", default=None, help="无头渲染工作台并输出每个视图的 PNG")
    args = parser.parse_args()
    if args.smoke:
        sys.exit(smoke(args.file))
    if args.shot:
        sys.exit(shot(args.file, args.shot))
    app = make_app()
    win = MainWindow()
    win.show()
    if args.file:
        win.open_file(args.file)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
