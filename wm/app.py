"""Desktop shell. Talks to the pipeline only through the contract types, so the
algorithm track and this file can evolve independently.

    python -m wm.app [file]            interactive
    python -m wm.app --smoke [file]    offscreen self-check, no window shown
"""
from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QProgressBar, QPushButton, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QInputDialog, QMessageBox,
)

from . import documents, pipeline
from .batches import Batch, BatchWorker
from .contract import PageImage, PageResult, Thresholds

VIEW_MODES = ("结果", "原图", "掩码叠加", "分屏对比")
OPEN_FILTER = "可处理文档 (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"

DEFAULTS = Thresholds()

# (attr, 显示名, 最小值, 最大值, 步长, 单位, 影响阶段)
# 影响阶段决定重处理能否复用共享掩码：mask 类参数必须重新定位。
KNOBS: tuple[tuple[str, str, int, int, int, str, str], ...] = (
    ("lama_context_margin", "LaMa 上下文范围", 64, 512, 32, "px", "repair"),
    ("lama_tile_overlap", "LaMa 分块重叠", 0, 128, 8, "px", "repair"),
    ("lama_mask_dilate", "LaMa 掩码外扩", 0, 15, 1, "px", "repair"),
    ("close_px", "掩码笔画跨接", 0, 25, 1, "px", "mask"),
    ("mask_alpha_min", "水印判定阈值 α", 2, 30, 1, "%", "mask"),
)
# 掩码跨接宽度是 estimate_single_page 的入参而非 Thresholds 字段，默认值单独记。
KNOB_DEFAULTS = {"close_px": 11, "mask_alpha_min": int(round(DEFAULTS.mask_alpha_min * 100))}
KNOB_HINTS = {
    "lama_context_margin": "模型能看到的干净页面边界范围。水印铺满整页时无效果；"
                           "印章、角标等局部水印时越大越贴合周围版面。",
    "lama_tile_overlap": "512 分块之间的羽化宽度，只在修复区域超过 512px 时生效。",
    "lama_mask_dilate": "把重绘范围整体外扩若干像素。外扩到正文会直接触发保真度失败。",
    "close_px": "把被文字遮住的水印笔画接起来。过大会把相邻字形并入掩码。",
    "mask_alpha_min": "覆盖率低于该值的像素视为无水印。调低去得更狠，但误删正文风险上升。",
}


def _knob_default(attr: str) -> int:
    if attr in KNOB_DEFAULTS:
        return KNOB_DEFAULTS[attr]
    return int(getattr(DEFAULTS, attr))


def make_app() -> QApplication:
    """Qt's fallback font has no CJK glyphs, so every label renders as a box."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    font = QFont()
    font.setFamilies(["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
                      "WenQuanYi Micro Hei"])
    font.setPointSize(12)
    app.setFont(font)
    app.setStyleSheet(f"""
    QWidget {{ font-size: 15px; }}
    QMainWindow, QWidget {{ background: #f5f6f8; color: #1c1e21; }}
    QGroupBox {{
        border: 1px solid #d5d9e0; border-radius: 8px; margin-top: 18px;
        padding: 14px 12px 12px 12px; background: #ffffff; font-weight: 600;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
        left: 12px; top: 2px; padding: 0 6px; color: #444; }}
    QPushButton {{
        background: #ffffff; border: 1px solid #c9ced6; border-radius: 6px;
        padding: 8px 16px; font-weight: 600;
    }}
    QPushButton:hover {{ background: #eef3fb; border-color: #4a86d4; }}
    QPushButton:disabled {{ color: #a0a4ab; background: #f0f1f3; }}
    QPushButton:checked {{ background: #dbe8fb; border-color: #4a86d4; }}
    QListWidget, QTableWidget, QComboBox {{
        background: #ffffff; border: 1px solid #d5d9e0; border-radius: 6px; padding: 4px;
    }}
    QListWidget::item {{ padding: 8px 6px; }}
    QTableWidget::item {{ padding: 6px 8px; }}
    QHeaderView::section {{
        background: #eceff3; padding: 7px 8px; border: none;
        border-bottom: 1px solid #d5d9e0; font-weight: 600;
    }}
    QProgressBar {{
        border: 1px solid #d5d9e0; border-radius: 6px; height: 18px;
        text-align: center; background: #ffffff;
    }}
    QProgressBar::chunk {{ background: #4a86d4; border-radius: 5px; }}
    QSlider::groove:horizontal {{ height: 5px; background: #d8dce2; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: #4a86d4; border-radius: 3px; }}
    QSlider::handle:horizontal {{
        width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: #ffffff;
        border: 2px solid #4a86d4;
    }}
    QCheckBox {{ spacing: 8px; }}
    """)
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("离线智能去水印工作台")
        self.resize(1680, 980)
        self.batches: list[Batch] = []
        self.batch: Batch | None = None
        self._loading_params = False
        self._serial = 0
        self.pages: list[PageImage] = []
        self.results: list[PageResult] = []
        self._current = -1
        self._sliders: dict[str, QSlider] = {}

        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self.show_page)
        self.canvas = QLabel("打开 PDF 或图片开始")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.mode_box = QComboBox()
        self.mode_box.addItems(VIEW_MODES)
        self.mode_box.currentTextChanged.connect(self.render_canvas)
        self.open_btn = QPushButton("新增批次")
        self.open_btn.clicked.connect(self.open_file)
        self.run_btn = QPushButton("依次处理所有批次")
        # clicked 信号会携带 checked=False；直连 start_run 会被 only_page 收到，
        # 误入「重处理当前页」分支（self.pages[False] 即第 0 页）。必须用 lambda 隔参数。
        self.run_btn.clicked.connect(lambda: self.start_run())
        self.run_btn.setEnabled(False)
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.conditions = QLabel("判定：—")
        self.conditions.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m 张 · %p%")
        self.status = QLabel("就绪")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.batch_list = QListWidget()
        self.batch_list.currentRowChanged.connect(self.select_batch)
        left_layout.addWidget(QLabel("批次（同批水印样式应相同且位置对齐）"))
        left_layout.addWidget(self.batch_list, 1)
        batch_tools = QHBoxLayout()
        self.batch_edit_buttons = []
        for title, action in [("重命名", self.rename_batch), ("删除批次", self.delete_batch)]:
            button = QPushButton(title)
            button.clicked.connect(action)
            batch_tools.addWidget(button)
            self.batch_edit_buttons.append(button)
        left_layout.addLayout(batch_tools)
        self.append_btn = QPushButton("向当前批次添加图片")
        self.append_btn.clicked.connect(self.append_files)
        left_layout.addWidget(self.append_btn)
        self.group_hint = QLabel("每批按尺寸分组统计先验；单张组使用单图识别")
        self.group_hint.setWordWrap(True)
        left_layout.addWidget(self.group_hint)
        left_layout.addWidget(QLabel("当前批次图片"))
        left_layout.addWidget(self.page_list, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_btn)
        buttons.addWidget(self.run_btn)
        self.run_batch_btn = QPushButton("处理当前批次")
        self.run_batch_btn.clicked.connect(lambda: self.start_run(scope="batch"))
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.pause_queue)
        self.stop_btn = QPushButton("停止队列")
        self.stop_btn.clicked.connect(self.stop_queue)
        buttons.addWidget(self.run_batch_btn)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.export_btn)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("视图"))
        right_layout.addWidget(self.mode_box)
        self.param_box = self._build_param_box()
        right_layout.addWidget(self.param_box)
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
        splitter.setSizes([260, 1000, 470])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.worker: BatchWorker | None = None
        for slider in self._sliders.values():
            slider.valueChanged.connect(self.save_params)
        self.ocr_box.toggled.connect(self.save_params)
        self.style_box.currentTextChanged.connect(self.save_params)
        self.set_busy(False)

    # ---- 参数面板 -------------------------------------------------------

    def _build_param_box(self) -> QGroupBox:
        box = QGroupBox("处理参数")
        grid = QGridLayout(box)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        for row, (attr, label, lo, hi, step, unit, stage) in enumerate(KNOBS):
            caption = QLabel(f"{label} ({unit})")
            value = QLabel()
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setPageStep(step)
            s.setValue(_knob_default(attr))
            s.setToolTip(KNOB_HINTS.get(attr, ""))
            caption.setToolTip(KNOB_HINTS.get(attr, ""))

            def sync(v: int, _lab=value, _unit=unit) -> None:
                _lab.setText(f"{v} {_unit}")

            s.valueChanged.connect(sync)
            sync(s.value())
            grid.addWidget(caption, row, 0)
            grid.addWidget(s, row, 1)
            grid.addWidget(value, row, 2)
            self._sliders[attr] = s
        row = len(KNOBS)
        self.ocr_box = QCheckBox("OCR 水印定位与正文保护（需 OCR 模型）")
        self.ocr_box.setChecked(True)
        grid.addWidget(self.ocr_box, row, 0, 1, 3)
        row += 1
        grid.addWidget(QLabel("指定水印样式（可输中英文提示词）"), row, 0)
        self.style_box = QComboBox()
        self.style_box.setEditable(True)
        self.style_box.addItems(["自动识别", "点阵 dot grid", "平铺文字 tiled text",
                                 "居中大字 big center", "深色重字母 heavy dark", "角标 corner id"])
        self.style_box.setToolTip("自动=级联路由；指定后只走该样式通道，支持中英文提示词")
        grid.addWidget(self.style_box, row, 1, 1, 2)
        row += 1
        self.reprocess_btn = QPushButton("处理／重处理当前图片")
        self.reprocess_btn.clicked.connect(lambda: self.start_run(only_page=self._current))
        self.reprocess_btn.setEnabled(False)
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self.reset_params)
        grid.addWidget(self.reprocess_btn, row, 0, 1, 2)
        grid.addWidget(reset, row, 2)
        return box

    def current_thr(self) -> Thresholds:
        vals = {attr: s.value() for attr, s in self._sliders.items()}
        return Thresholds(
            mask_alpha_min=vals["mask_alpha_min"] / 100.0,
            lama_context_margin=vals["lama_context_margin"],
            lama_tile_overlap=vals["lama_tile_overlap"],
            lama_mask_dilate=vals["lama_mask_dilate"],
        )

    def current_close_px(self) -> int:
        return self._sliders["close_px"].value()

    def current_style(self) -> str | None:
        from .styles import parse_style_prompt
        return parse_style_prompt(self.style_box.currentText())

    def reset_params(self) -> None:
        for attr, s in self._sliders.items():
            s.setValue(_knob_default(attr))
        self.ocr_box.setChecked(True)

    def save_params(self, *_args) -> None:
        if self._loading_params or self.batch is None:
            return
        params = {attr: slider.value() for attr, slider in self._sliders.items()}
        style = self.current_style()
        if (any(self.batch.params.get(k) != params[k] for k in ("mask_alpha_min", "close_px"))
                or self.batch.use_ocr != self.ocr_box.isChecked()
                or self.batch.style != style):
            self.batch.invalidate()
        self.batch.params = params
        self.batch.use_ocr = self.ocr_box.isChecked()
        self.batch.style = style

    def select_batch(self, row: int) -> None:
        self.batch = self.batches[row] if 0 <= row < len(self.batches) else None
        self.pages = self.batch.pages if self.batch else []
        self.results = list(self.batch.results.values()) if self.batch else []
        self._loading_params = True
        for attr, slider in self._sliders.items():
            slider.setValue(self.batch.params.get(attr, _knob_default(attr))
                            if self.batch else _knob_default(attr))
        self.ocr_box.setChecked(self.batch.use_ocr if self.batch else True)
        self.style_box.setCurrentText((self.batch.style or "自动识别") if self.batch else "自动识别")
        self._loading_params = False
        self.page_list.clear()
        if self.batch:
            sizes = {}
            for page in self.pages:
                sizes[page.image.shape] = sizes.get(page.image.shape, 0) + 1
            self.group_hint.setText("尺寸分组：" + "；".join(
                f"{shape[1]}×{shape[0]} · {count} 张" + ("（单图识别）" if count == 1 else "（跨页先验）")
                for shape, count in sizes.items()))
            for i, page in enumerate(self.pages):
                self.page_list.addItem(f"{i+1}. {page.doc_id} · {self.batch.states.get(i, '未处理')}")
        self.page_list.setCurrentRow(0)
        self.show_page(0 if self.pages else -1)
        self.set_busy(bool(self.worker and self.worker.isRunning()))

    def rename_batch(self) -> None:
        if not self.batch or (self.worker and self.worker.isRunning()):
            return
        name, ok = QInputDialog.getText(self, "重命名批次", "批次名称", text=self.batch.name)
        if ok and name.strip():
            self.batch.name = name.strip()
            self.refresh_batches()

    def delete_batch(self) -> None:
        if not self.batch or (self.worker and self.worker.isRunning()):
            return
        if self.batch.dirty and QMessageBox.question(
                self, "删除批次", "此批次有未导出的结果，仍要删除吗？") != QMessageBox.Yes:
            return
        row = self.batch_list.currentRow()
        self.batches.pop(row)
        self.refresh_batches()
        self.batch_list.setCurrentRow(min(row, len(self.batches)-1))
        self.select_batch(self.batch_list.currentRow())

    def refresh_batches(self) -> None:
        row = self.batch_list.currentRow()
        self.batch_list.blockSignals(True)
        self.batch_list.clear()
        for batch in self.batches:
            self.batch_list.addItem(f"{batch.name} · {len(batch.pages)} 张 · {len(batch.results)} 张有结果")
        self.batch_list.setCurrentRow(row)
        self.batch_list.blockSignals(False)

    def open_file(self, path: str | Path | None = None) -> None:
        if self.worker and self.worker.isRunning():
            return
        files = [str(path)] if path else QFileDialog.getOpenFileNames(
            self, "新增批次：选择水印相同的图片或文档", "data", OPEN_FILTER)[0]
        if not files:
            return
        try:
            pages = documents.load_many(files)
            if not pages:
                raise ValueError("所选文档没有页面")
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self._serial += 1
        batch = Batch(f"批次 {self._serial} · {Path(files[0]).stem}", pages)
        batch.params = {attr: _knob_default(attr) for attr in self._sliders}
        self.batches.append(batch)
        self.refresh_batches()
        self.batch_list.setCurrentRow(len(self.batches)-1)
        self.progress.setMaximum(len(pages))
        self.progress.setValue(0)
        self.status.setText(f"已新增 {batch.name}，共 {len(pages)} 张")

    def append_files(self) -> None:
        if not self.batch or (self.worker and self.worker.isRunning()):
            return
        files = QFileDialog.getOpenFileNames(self, "添加到当前批次", "data", OPEN_FILTER)[0]
        if not files:
            return
        try:
            pages = documents.load_many(files)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        for page in pages:
            page.page_index = len(self.batch.pages)
            self.batch.pages.append(page)
        self.batch.invalidate()
        self.refresh_batches()
        self.select_batch(self.batch_list.currentRow())

    def export_results(self) -> None:
        scope, ok = QInputDialog.getItem(self, "导出范围", "选择导出范围",
                                         ["当前图片", "当前批次", "所有批次"], 0, False)
        if not ok:
            return
        selections = [(b, list(range(len(b.pages)))) for b in self.batches] if scope == "所有批次" else (
            [(self.batch, [self._current] if scope == "当前图片" else list(range(len(self.pages))))]
            if self.batch else [])
        if not any(i in b.results for b, indices in selections for i in indices):
            self.status.setText("所选范围没有可导出的结果")
            return
        chosen = QFileDialog.getExistingDirectory(self, "选择结果导出目录", "out")
        if not chosen:
            return
        root = Path(chosen) / ("batches_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        try:
            for batch, indices in selections:
                safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in batch.name)[:80]
                target = root / f"{safe}_{batch.uid[:8]}"
                results = [batch.results[i] for i in indices if i in batch.results]
                pipeline.write_results(results, target)
                (target / "queue.json").write_text(json.dumps([
                    dict(page=batch.pages[i].key, state=batch.states.get(i, "未处理"),
                         error=batch.errors.get(i)) for i in indices], ensure_ascii=False, indent=2), "utf-8")
                batch.dirty.difference_update(i for i in indices if i in batch.results)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.status.setText(f"已导出 → {root}")

    def set_busy(self, busy: bool) -> None:
        self.param_box.setEnabled(not busy)
        for button in self.batch_edit_buttons:
            button.setEnabled(not busy and self.batch is not None)
        self.open_btn.setEnabled(not busy)
        self.append_btn.setEnabled(not busy and self.batch is not None)
        self.run_btn.setEnabled(not busy and bool(self.batches))
        self.run_batch_btn.setEnabled(not busy and bool(self.pages))
        self.reprocess_btn.setEnabled(not busy and bool(self.pages))
        for slider in self._sliders.values():
            slider.setEnabled(not busy)
        self.ocr_box.setEnabled(not busy)
        self.style_box.setEnabled(not busy)
        self.pause_btn.setEnabled(busy)
        self.stop_btn.setEnabled(busy)
        self.export_btn.setEnabled(not busy and any(b.results for b in self.batches))

    def start_run(self, only_page: int | None = None, scope: str = "all") -> None:
        if not self.batches or (self.worker and self.worker.isRunning()):
            return
        self.save_params()
        batches = self.batches if scope == "all" and only_page is None else [self.batch]
        jobs = []
        for batch in batches:
            if batch is None:
                continue
            indices = [only_page] if only_page is not None else list(range(len(batch.pages)))
            indices = [i for i in indices if 0 <= i < len(batch.pages)]
            if not indices:
                continue
            vals = batch.params
            thr = Thresholds(mask_alpha_min=vals["mask_alpha_min"] / 100,
                             lama_context_margin=vals["lama_context_margin"],
                             lama_tile_overlap=vals["lama_tile_overlap"],
                             lama_mask_dilate=vals["lama_mask_dilate"])
            jobs.append((batch, indices, thr, vals["close_px"], batch.use_ocr, batch.style))
        if not jobs:
            return
        self.worker = BatchWorker(jobs, self)
        self.worker.update.connect(self.queue_update)
        self.worker.finished.connect(self.on_finished)
        self.progress.setRange(0, self.worker.total)
        self.progress.setValue(0)
        self.pause_btn.setChecked(False)
        self.set_busy(True)
        self.status.setText(f"开始处理，共 {self.worker.total} 张；已有结果将在各张重跑后更新")
        self.worker.start()

    def pause_queue(self, paused: bool) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.pause(paused)
            self.pause_btn.setText("继续" if paused else "暂停")
            self.status.setText("已请求暂停，当前图片完成后生效" if paused else "继续处理")

    def stop_queue(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status.setText("已请求停止，当前图片完成后生效")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def queue_update(self, payload: dict) -> None:
        if payload["kind"] == "paused":
            self.status.setText(f"已暂停 · {payload['done']}/{payload['total']} 张")
            return
        batch = next(b for b in self.batches if b.uid == payload["batch"])
        index = payload["index"]
        if payload["kind"] == "stage":
            self.status.setText(f"{batch.name} · 第 {index+1} 张 · {payload['message']}")
            return
        result = payload["result"]
        if result is not None:
            batch.results[index] = result
            batch.errors.pop(index, None)
            batch.dirty.add(index)
            state = "质量未通过" if not result.verdict.passed else (
                "待复核" if result.verdict.review_needed else "通过")
        else:
            # A failed rerun must not leave an old result looking current.
            batch.results.pop(index, None)
            batch.dirty.discard(index)
            batch.errors[index] = payload["error"]
            state = "运行失败"
        batch.states[index] = state
        self.progress.setValue(payload["done"])
        self.status.setText(f"{batch.name} · 第 {index+1} 张 · {state} · "
                            f"总进度 {payload['done']}/{payload['total']}" +
                            (f" · {payload['error']}" if payload["error"] else ""))
        self.refresh_batches()
        if batch is self.batch:
            self.results = list(batch.results.values())
            self.page_list.item(index).setText(f"{index+1}. {batch.pages[index].doc_id} · {state}")
            self.page_list.item(index).setBackground(
                Qt.darkRed if state in ("运行失败", "质量未通过") else
                Qt.darkYellow if state == "待复核" else Qt.darkGreen)
            if self._current == index:
                self.show_page(index)

    def on_finished(self) -> None:
        worker = self.worker
        self.set_busy(False)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("暂停")
        self.pause_btn.blockSignals(False)
        counts = {}
        for state in worker.outcomes:
            counts[state] = counts.get(state, 0) + 1
        summary = " · ".join(f"{state} {count}" for state, count in counts.items())
        self.status.setText(f"{'已停止' if worker.stopped else '完成'} · 本次 {worker.done}/{worker.total} 张 · {summary}")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "队列运行中", "请先停止队列，等待当前图片完成后再关闭。")
            event.ignore()
            return
        if any(b.dirty for b in self.batches):
            answer = QMessageBox.question(self, "未导出的结果", "仍有未导出的结果。是否放弃并关闭？",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()

    def show_page(self, row: int) -> None:
        self._current = row
        self.metrics.setRowCount(0)
        self.conditions.setText("判定：—")
        result = self.batch.results.get(row) if self.batch else None
        if result is not None:
            self._fill_metrics(result)
        elif self.batch and row in self.batch.errors:
            self.conditions.setText("运行失败：" + self.batch.errors[row])
        if not (0 <= row < len(self.pages)):
            self.canvas.clear()
            self.canvas.setText("新增批次并上传同水印图片开始")
        self.render_canvas()

    def _fill_metrics(self, result: PageResult) -> None:
        report = result.verdict
        rows = [("引擎", str(report.engine)), ("掩码来源", str(result.mask.source)),
                ("定位方式", {"repeated_document_overlay": "重复文字与透明度估计",
                              "repeated_oblique_text": "重复斜向文字",
                              "single_periodic_overlay": "单图点阵与透明度估计",
                              "single_periodic_document": "单图文档点阵与灰度分布估计",
                              "single_repeated_paper": "单图整页重复文字定位",
                              "transparent_pdf_text": "PDF 透明文字对象移除",
                              "single_chromatic_oblique": "单图色彩残差与大字定位",
                              "single_coloured_document_overlay": "单图文档色差与透明度估计",
                              "crosspage": "跨页统计"}.get(result.mask.method, "灰度与颜色")),
                ("耗时 ms", report.elapsed_ms), ("阶段", report.stage_ms)]
        rows += [(k, f"{v:.4f}" if isinstance(v, float) else v)
                 for k, v in sorted(report.metrics.items())]
        self.metrics.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(str(k)))
            self.metrics.setItem(i, 1, QTableWidgetItem(str(v)))
        flags = "  ".join(
            f"{'✔' if ok else '✘'}{name}" for name, ok in report.conditions.items())
        review = "\n处理效果待人工复核，自动检查可能漏掉水印" if report.review_needed else ""
        self.conditions.setText(f"自动检查：{flags}{review}\n异常：{[str(f) for f in report.failures] or '无'}")

    def render_canvas(self) -> None:
        row = self._current
        if not (0 <= row < len(self.pages)):
            return
        source = self.pages[row].image
        result = self.batch.results.get(row) if self.batch else None
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
        # 高分屏（Retina）下 canvas.size() 是逻辑像素，直接缩放会被系统放大 2 倍导致模糊；
        # 按物理像素缩放并标记 devicePixelRatio，Qt 即 1:1 贴图。
        dpr = self.canvas.devicePixelRatioF()
        scaled = pixmap.scaled(self.canvas.size() * dpr, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(dpr)
        self.canvas.setPixmap(scaled)


def smoke(path: str | None) -> int:
    app = make_app()
    win = MainWindow()
    win.show()
    win.open_file(path or str(Path(__file__).resolve().parents[1] / "data/watermark_regression_min_v1/inputs/core/core_001.png"))
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
    win.open_file(path or str(Path(__file__).resolve().parents[1] / "data/watermark_regression_min_v1/inputs/core/core_025.pdf"))
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
