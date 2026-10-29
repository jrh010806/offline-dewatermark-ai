"""Desktop shell. Talks to the pipeline only through the contract types, so the
algorithm track and this file can evolve independently.

    python -m wm.app [file]            interactive
    python -m wm.app --smoke [file]    offscreen self-check, no window shown
"""
from __future__ import annotations

import argparse
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QLocale, QTranslator, QLibraryInfo
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QProgressBar, QPushButton, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QInputDialog, QMessageBox,
    QAbstractItemView, QScrollArea, QButtonGroup, QSpinBox,
)

from . import documents, pipeline
from .batches import Batch, BatchWorker
from .contract import PageImage, PageResult, Thresholds
from .viewer import ImageViewer
from .export_review import overlay_mask
from .assist_worker import AssistWorker
from .selection_edit import SelectionEdit

VIEW_MODES = ("拖拽对比", "处理结果", "原始图片", "水印选区")
STYLE_LABELS = {None: "自动识别", "dot_grid": "点阵水印", "tiled_text": "平铺文字",
                "big_center": "居中大字", "heavy_dark": "深色文字", "corner_id": "角标水印"}
OPEN_FILTER = "可处理文档 (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"

DEFAULTS = Thresholds()

# (attr, 显示名, 最小值, 最大值, 步长, 单位, 影响阶段)
# 影响阶段决定重处理能否复用共享掩码：mask 类参数必须重新定位。
KNOBS: tuple[tuple[str, str, int, int, int, str, str], ...] = (
    ("lama_context_margin", "修复参考范围", 64, 512, 32, "像素", "repair"),
    ("lama_tile_overlap", "修复分块重叠", 0, 128, 8, "像素", "repair"),
    ("lama_mask_dilate", "水印选区外扩", 0, 15, 1, "像素", "repair"),
    ("close_px", "水印笔画连接", 0, 25, 1, "像素", "mask"),
    ("mask_alpha_min", "水印判定阈值", 2, 30, 1, "%", "mask"),
)
# 掩码跨接宽度是 estimate_single_page 的入参而非 Thresholds 字段，默认值单独记。
KNOB_DEFAULTS = {"close_px": 11, "mask_alpha_min": int(round(DEFAULTS.mask_alpha_min * 100))}
KNOB_HINTS = {
    "lama_context_margin": "模型能看到的干净页面边界范围。水印铺满整页时无效果；"
                           "印章、角标等局部水印时越大越贴合周围版面。",
    "lama_tile_overlap": "512 分块之间的羽化宽度，只在修复区域超过 512 像素 时生效。",
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
    QLocale.setDefault(QLocale(QLocale.Chinese, QLocale.China))
    if not hasattr(app, "_chinese_translator"):
        translator = QTranslator(app)
        if translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            app.installTranslator(translator)
        app._chinese_translator = translator
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
                      "WenQuanYi Micro Hei"])
    font.setPointSize(12)
    app.setFont(font)
    app.setStyleSheet("""
    QWidget { font-size: 13px; color: #253c35; }
    QMainWindow, QWidget#root { background: #f1f4f2; }
    QWidget#panel { background: #ffffff; border-radius: 12px; }
    QLabel { background: transparent; }
    QLabel#heading { font-size: 23px; font-weight: 700; color: #163e31; }
    QLabel#section { font-size: 15px; font-weight: 600; }
    QLabel#muted { color: #7b8983; font-size: 12px; }
    QLabel#notice { background: #f3f6f4; border-radius: 8px; padding: 12px; }
    QLabel#local { color: #23785a; background: #e3f2e9; border-radius: 12px; padding: 7px 14px; }
    QGroupBox { border: 1px solid #e1e8e3; border-radius: 10px; margin-top: 20px;
        padding: 10px 10px 10px; background: #ffffff; font-weight: 600; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QPushButton { background: #ffffff; border: 1px solid #d9e2dc; border-radius: 8px;
        padding: 9px 12px; font-weight: 600; }
    QPushButton:hover { background: #edf6f0; border-color: #559b7c; }
    QPushButton:pressed, QPushButton:checked { background: #dceee2; }
    QPushButton#primary { background: #257853; color: white; border-color: #257853; }
    QPushButton#primary:hover { background: #1b6343; }
    QPushButton:disabled, QPushButton#primary:disabled { color: #a3afa8; background: #eef2ef; border-color: #e5ebe7; }
    QWidget#viewSwitch { background: #e6ede8; border: 1px solid #dce5df; border-radius: 10px; }
    QWidget#viewSwitch QPushButton { background: transparent; color: #64776c;
        border: 1px solid transparent; border-radius: 7px; padding: 7px 10px; font-weight: 500; }
    QWidget#viewSwitch QPushButton:hover { background: #f6f9f7; color: #25543a; }
    QWidget#viewSwitch QPushButton:checked { background: #257853; color: #ffffff; font-weight: 600; }
    QWidget#viewSwitch QPushButton:focus { border: 1px solid #83b79a; }
    QListWidget, QTableWidget, QComboBox, QLineEdit { background: #ffffff;
        border: 1px solid #e1e8e3; border-radius: 8px; padding: 5px; }
    QComboBox { min-height: 24px; }
    QComboBox QAbstractItemView { selection-background-color: #dfefe5; selection-color: #174c33; }
    QListWidget::item { padding: 10px 8px; border-radius: 6px; margin: 2px; }
    QListWidget::item:selected { background: #e1f0e6; color: #174c33; }
    QListWidget::item:hover { background: #f0f6f2; }
    QTableWidget { gridline-color: #eef2ef; }
    QTableWidget::item { padding: 5px; }
    QHeaderView::section { background: #f3f6f4; padding: 8px; border: none;
        border-bottom: 1px solid #e1e8e3; font-weight: 600; }
    QProgressBar { border: none; border-radius: 5px; max-height: 8px;
        background: #dfe8e2; }
    QProgressBar::chunk { background: #39875f; border-radius: 4px; }
    QSlider::groove:horizontal { height: 4px; background: #dce5df; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: #438c68; border-radius: 2px; }
    QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0;
        border-radius: 8px; background: white; border: 2px solid #438c68; }
    QCheckBox { spacing: 7px; }
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical { background: #f1f4f2; width: 8px; margin: 0; }
    QScrollBar::handle:vertical { background: #c6d3ca; border-radius: 4px; min-height: 30px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
    QComboBox::drop-down { border: none; width: 22px; }
    QComboBox::down-arrow { image: url("@CHEVRON@"); width: 16px; height: 16px; }
    QSplitter::handle { background: transparent; width: 10px; }
    QToolTip { background: #ffffff; color: #253c35; border: 1px solid #d9e2dc; padding: 6px; }
    """.replace("@CHEVRON@", (Path(__file__).parent / "assets" / "chevron-down.svg").as_posix()))
    return app


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("离线智能去水印工作台")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 740)
        self.batches: list[Batch] = []
        self.batch: Batch | None = None
        self._loading_params = False
        self._serial = 0
        self.pages: list[PageImage] = []
        self.results: list[PageResult] = []
        self._current = -1
        self._sliders: dict[str, QSlider] = {}
        self.assist_worker = None
        self._assist_busy = False
        self._assist_pending = None
        self._assist_context = None
        self._assist_payload = None
        self._assist_undo = {}
        self._selection_edit = None
        self._ui_busy = False

        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self.show_page)
        self.canvas = ImageViewer()
        self.canvas.rectangleFinished.connect(self.find_assisted_strokes)
        self.canvas.brushStarted.connect(self._begin_brush_stroke)
        self.canvas.brushSegment.connect(self._paint_brush_segment)
        self.canvas.brushFinished.connect(self._end_brush_stroke)
        self.canvas.undoRequested.connect(self.undo_selection_stroke)
        self.canvas.redoRequested.connect(self.redo_selection_stroke)
        self.canvas.toolRequested.connect(self.set_brush_tool)
        self.page_list.setTextElideMode(Qt.ElideMiddle)
        self.page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view_mode = VIEW_MODES[0]
        self.view_switch = QWidget()
        self.view_switch.setObjectName("viewSwitch")
        self.view_switch.setAccessibleName("预览视图")
        view_layout = QHBoxLayout(self.view_switch)
        view_layout.setContentsMargins(4, 4, 4, 4)
        view_layout.setSpacing(2)
        self.view_group = QButtonGroup(self)
        self.view_buttons = {}
        for mode in VIEW_MODES:
            button = QPushButton(mode)
            button.setCheckable(True)
            button.setChecked(mode == self._view_mode)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, mode=mode: self.set_view_mode(mode))
            self.view_group.addButton(button)
            view_layout.addWidget(button)
            self.view_buttons[mode] = button
        self.open_btn = QPushButton("＋ 新增批次")
        self.open_btn.clicked.connect(self.open_file)
        self.run_btn = QPushButton("自动处理全部批次")
        # clicked 信号会携带 checked=False；直连 start_run 会被 only_page 收到，
        # 误入「重处理当前页」分支（self.pages[False] 即第 0 页）。必须用 lambda 隔参数。
        self.run_btn.clicked.connect(lambda: self.start_run())
        self.run_btn.setEnabled(False)
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics.verticalHeader().hide()
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics.setShowGrid(False)
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.conditions = QLabel("判定：—")
        self.conditions.setWordWrap(True)
        self.conditions.setObjectName("notice")
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m 张 · %p%")
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        self.status = QLabel("就绪")

        def label(text, kind="section"):
            widget = QLabel(text)
            widget.setObjectName(kind)
            return widget

        left = QWidget()
        left.setObjectName("panel")
        left.setMinimumWidth(230)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 18, 16, 18)
        left_layout.setSpacing(12)
        self.batch_list = QListWidget()
        self.batch_list.setTextElideMode(Qt.ElideMiddle)
        self.batch_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.batch_list.currentRowChanged.connect(self.select_batch)
        left_layout.addWidget(label("批次管理"))
        hint = label("同批次图片的水印样式与位置应一致", "muted")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        self.open_btn.setObjectName("primary")
        left_layout.addWidget(self.open_btn)
        left_layout.addWidget(self.batch_list, 1)
        batch_tools = QHBoxLayout()
        self.batch_edit_buttons = []
        for title, action in [("重命名", self.rename_batch), ("删除批次", self.delete_batch)]:
            button = QPushButton(title)
            button.clicked.connect(action)
            batch_tools.addWidget(button)
            self.batch_edit_buttons.append(button)
        left_layout.addLayout(batch_tools)
        self.append_btn = QPushButton("＋ 添加图片到当前批次")
        self.append_btn.clicked.connect(self.append_files)
        left_layout.addWidget(self.append_btn)
        self.group_hint = label("按图片尺寸自动分组，支持单图识别", "muted")
        self.group_hint.setWordWrap(True)
        left_layout.addWidget(self.group_hint)
        left_layout.addWidget(label("图片列表"))
        left_layout.addWidget(self.page_list, 2)

        self.run_batch_btn = QPushButton("自动处理当前批次")
        self.run_batch_btn.setObjectName("primary")
        self.run_batch_btn.clicked.connect(lambda: self.start_run(scope="batch"))
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.pause_queue)
        self.stop_btn = QPushButton("停止队列")
        self.stop_btn.clicked.connect(self.stop_queue)

        right = QWidget()
        right.setObjectName("panel")
        right.setMinimumWidth(330)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(8)
        right_layout.addWidget(label("处理设置"))
        self.param_box = self._build_param_box()
        right_layout.addWidget(self.param_box)
        right_layout.addWidget(label("质量检查"))
        self.metrics.setMinimumHeight(150)
        right_layout.addWidget(self.metrics, 1)
        right_layout.addWidget(self.conditions)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        scroll.setWidget(right)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 0, 4, 0)
        center_layout.setSpacing(12)
        preview_bar = QHBoxLayout()
        preview_bar.addWidget(label("效果预览"))
        preview_bar.addStretch()
        preview_bar.addWidget(self.view_switch)
        center_layout.addLayout(preview_bar)
        self.preview_info = label("选择图片后即可预览", "muted")
        center_layout.addWidget(self.preview_info)
        assist_bar = QHBoxLayout()
        self.assist_select_btn = QPushButton("编辑水印选区")
        self.assist_select_btn.setToolTip("在原图上添加或擦除完整选区；点击处理才生效，切图丢弃未处理编辑。")
        self.assist_select_btn.setCheckable(True)
        self.assist_select_btn.toggled.connect(self.begin_assisted_selection)
        self.assist_confirm_btn = QPushButton("按当前选区处理")
        self.assist_confirm_btn.setObjectName("primary")
        self.assist_confirm_btn.clicked.connect(self.confirm_assisted_repair)
        self.assist_cancel_btn = QPushButton("取消编辑")
        self.assist_cancel_btn.clicked.connect(self.cancel_assisted_selection)
        self.assist_undo_btn = QPushButton("撤销本次处理")
        self.assist_undo_btn.clicked.connect(self.undo_assisted_repair)
        for button in (self.assist_select_btn, self.assist_confirm_btn,
                       self.assist_cancel_btn, self.assist_undo_btn):
            assist_bar.addWidget(button)
        assist_bar.addStretch()
        center_layout.addLayout(assist_bar)
        self.brush_tools = QWidget()
        tools_layout = QHBoxLayout(self.brush_tools)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        self.add_brush_btn = QPushButton("添加 B")
        self.erase_brush_btn = QPushButton("擦除 E")
        self.brush_group = QButtonGroup(self)
        for button, tool in ((self.add_brush_btn, 'add'), (self.erase_brush_btn, 'erase')):
            button.setCheckable(True)
            self.brush_group.addButton(button)
            button.clicked.connect(lambda _checked=False, tool=tool: self.set_brush_tool(tool))
            tools_layout.addWidget(button)
        self.add_brush_btn.setChecked(True)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 300)
        self.brush_size.setValue(12)
        self.brush_size.setSuffix(" 像素")
        self.brush_size.setToolTip("原图中的笔刷直径；缩放画面不会改变实际选中宽度")
        self.brush_size.valueChanged.connect(self._brush_size_changed)
        tools_layout.addWidget(QLabel("笔刷"))
        tools_layout.addWidget(self.brush_size)
        self.stroke_undo_btn = QPushButton("撤销笔画")
        self.stroke_redo_btn = QPushButton("重做")
        self.stroke_undo_btn.clicked.connect(self.undo_selection_stroke)
        self.stroke_redo_btn.clicked.connect(self.redo_selection_stroke)
        tools_layout.addWidget(self.stroke_undo_btn)
        tools_layout.addWidget(self.stroke_redo_btn)
        tools_layout.addStretch()
        center_layout.addWidget(self.brush_tools)
        self.brush_tools.hide()
        zoom_bar = QHBoxLayout()
        self.zoom_out_btn = QPushButton("−")
        self.zoom_in_btn = QPushButton("＋")
        self.fit_btn = QPushButton("适应画布")
        self.actual_size_btn = QPushButton("原始大小")
        self.zoom_label = QLabel("100%")
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_at(self.canvas.image_rect().center(), 1/1.5))
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_at(self.canvas.image_rect().center(), 1.5))
        self.fit_btn.clicked.connect(self.canvas.fit_image)
        self.actual_size_btn.clicked.connect(lambda: self.canvas.zoom_at(
            self.canvas.image_rect().center(), 1/self.canvas.image_scale()))
        self.canvas.zoomChanged.connect(lambda scale: self.zoom_label.setText(f"{scale:.0%}"))
        for widget in (self.zoom_out_btn, self.zoom_label, self.zoom_in_btn, self.fit_btn, self.actual_size_btn):
            zoom_bar.addWidget(widget)
        self.magnifier_box = QCheckBox("画笔放大镜")
        self.magnifier_box.setChecked(True)
        self.magnifier_box.toggled.connect(self._set_magnifier)
        zoom_bar.addStretch()
        zoom_bar.addWidget(self.magnifier_box)
        center_layout.addLayout(zoom_bar)
        self.selection_notice = label("", "muted")
        self.selection_notice.setWordWrap(True)
        center_layout.addWidget(self.selection_notice)
        center_layout.addWidget(self.canvas, 1)
        self.preview_hint = label("先导入图片，再开始处理", "muted")
        self.preview_hint.setAlignment(Qt.AlignCenter)
        self.preview_hint.setWordWrap(True)
        center_layout.addWidget(self.preview_hint)
        splitter.addWidget(center)
        splitter.addWidget(scroll)
        splitter.setSizes([270, 880, 390])
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(22, 18, 22, 16)
        root_layout.setSpacing(16)
        heading = QHBoxLayout()
        heading.addWidget(label("离线智能去水印", "heading"))
        heading.addStretch()
        heading.addWidget(label("● 本地处理 · 文件无需上传", "local"))
        root_layout.addLayout(heading)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.run_batch_btn)
        toolbar.addWidget(self.run_btn)
        toolbar.addWidget(self.pause_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.export_btn)
        root_layout.addLayout(toolbar)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.worker: BatchWorker | None = None
        for slider in self._sliders.values():
            slider.valueChanged.connect(self.save_params)
        self.ocr_box.toggled.connect(self.save_params)
        self.recall_box.toggled.connect(self.save_params)
        self.style_box.currentTextChanged.connect(self.save_params)
        self.set_busy(False)

    # ---- 参数面板 -------------------------------------------------------

    def _build_param_box(self) -> QGroupBox:
        box = QGroupBox("处理参数")
        grid = QGridLayout(box)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        for row, (attr, label, lo, hi, step, unit, stage) in enumerate(KNOBS):
            caption = QLabel(label)
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
            grid.addWidget(caption, row * 2, 0, 1, 2)
            grid.addWidget(value, row * 2, 2)
            grid.addWidget(s, row * 2 + 1, 0, 1, 3)
            self._sliders[attr] = s
        row = len(KNOBS) * 2
        self.ocr_box = QCheckBox("文字识别")
        self.ocr_box.setChecked(True)
        grid.addWidget(self.ocr_box, row, 0, 1, 3)
        row += 1
        self.recall_box = QCheckBox("补全水印笔画（允许正文交叠）")
        self.recall_box.setChecked(True)
        self.recall_box.setToolTip("通过颜色和重复字形补全笔画；文字框和重复行只用于搜索，不作为修复选区。外扩由上方参数控制。")
        grid.addWidget(self.recall_box, row, 0, 1, 3)
        row += 1
        grid.addWidget(QLabel("水印样式"), row, 0)
        self.style_box = QComboBox()
        self.style_box.setEditable(True)
        self.style_box.addItems(list(STYLE_LABELS.values()))
        self.style_box.setToolTip("自动识别会依次检查水印类型；也可选择样式或输入中文描述")
        grid.addWidget(self.style_box, row, 1, 1, 2)
        row += 1
        self.reprocess_btn = QPushButton("处理／重处理当前图片")
        self.reprocess_btn.clicked.connect(lambda: self.start_run(only_page=self._current))
        self.reprocess_btn.setEnabled(False)
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self.reset_params)
        grid.addWidget(self.reprocess_btn, row, 0, 1, 2)
        grid.addWidget(reset, row, 2)
        self.auto_redetect_btn = QPushButton("重新自动识别当前图片")
        self.auto_redetect_btn.setToolTip("重新检测将替换该页人工选区和结果；需要保留时请先导出。")
        self.auto_redetect_btn.clicked.connect(lambda: self.start_run(
            only_page=self._current, force_automatic=True))
        grid.addWidget(self.auto_redetect_btn, row + 1, 0, 1, 3)
        self.auto_redetect_btn.hide()
        return box

    def current_thr(self) -> Thresholds:
        vals = {attr: s.value() for attr, s in self._sliders.items()}
        return Thresholds(
            mask_alpha_min=vals["mask_alpha_min"] / 100.0,
            lama_context_margin=vals["lama_context_margin"],
            lama_tile_overlap=vals["lama_tile_overlap"],
            lama_mask_dilate=vals["lama_mask_dilate"],
            watermark_recall_priority=self.recall_box.isChecked(),
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
        self.recall_box.setChecked(True)
        self.style_box.setCurrentText("自动识别")

    def save_params(self, *_args) -> None:
        if self._loading_params or self.batch is None:
            return
        if self._selection_edit is None:
            self._clear_assisted_preview()
        params = {attr: slider.value() for attr, slider in self._sliders.items()}
        style = self.current_style()
        if (any(self.batch.params.get(k) != params[k] for k in ("mask_alpha_min", "close_px"))
                or self.batch.use_ocr != self.ocr_box.isChecked()
                or self.batch.recall_priority != self.recall_box.isChecked()
                or self.batch.style != style):
            self.batch.invalidate()
        self.batch.params = params
        self.batch.use_ocr = self.ocr_box.isChecked()
        self.batch.recall_priority = self.recall_box.isChecked()
        self.batch.style = style
        self.render_canvas()

    def select_batch(self, row: int) -> None:
        self._clear_assisted_preview()
        self.batch = self.batches[row] if 0 <= row < len(self.batches) else None
        self.pages = self.batch.pages if self.batch else []
        self.results = list(self.batch.results.values()) if self.batch else []
        self._loading_params = True
        for attr, slider in self._sliders.items():
            slider.setValue(self.batch.params.get(attr, _knob_default(attr))
                            if self.batch else _knob_default(attr))
        self.ocr_box.setChecked(self.batch.use_ocr if self.batch else True)
        self.recall_box.setChecked(self.batch.recall_priority if self.batch else True)
        self.style_box.setCurrentText(STYLE_LABELS.get(self.batch.style if self.batch else None, "自动识别"))
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
                self.page_list.addItem(self._page_item(i))
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
            item = QListWidgetItem(f"{batch.name}\n{len(batch.pages)} 张图片 · {len(batch.results)} 张有结果")
            item.setToolTip(batch.name)
            self.batch_list.addItem(item)
        self.batch_list.setCurrentRow(row)
        self.batch_list.blockSignals(False)

    def open_file(self, path: str | Path | None = None) -> None:
        if self.worker and self.worker.isRunning():
            return
        files = [str(path)] if path else QFileDialog.getOpenFileNames(
            self, "新增批次：选择水印相同的图片或文档", "data", OPEN_FILTER, options=QFileDialog.DontUseNativeDialog)[0]
        if not files:
            return
        try:
            pages = documents.load_many(files)
            if not pages:
                raise ValueError("所选文档没有页面")
        except Exception:
            logging.getLogger(__name__).exception("文件操作失败")
            QMessageBox.warning(self, "导入失败", "无法读取所选文件，请检查文件格式、内容或访问权限。")
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
        files = QFileDialog.getOpenFileNames(self, "添加到当前批次", "data", OPEN_FILTER, options=QFileDialog.DontUseNativeDialog)[0]
        if not files:
            return
        try:
            pages = documents.load_many(files)
        except Exception:
            logging.getLogger(__name__).exception("文件操作失败")
            QMessageBox.warning(self, "导入失败", "无法读取所选文件，请检查文件格式、内容或访问权限。")
            return
        for page in pages:
            page.page_index = len(self.batch.pages)
            self.batch.pages.append(page)
        self.batch.invalidate()
        self.refresh_batches()
        self.select_batch(self.batch_list.currentRow())

    def export_results(self) -> None:
        if self._selection_edit is not None and self._selection_edit.dirty:
            self.status.setText("选区尚未处理：请先按当前选区处理，或取消编辑后导出已有结果。")
            return
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
        chosen = QFileDialog.getExistingDirectory(
            self, "选择结果导出目录", "out", QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
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
        except Exception:
            logging.getLogger(__name__).exception("文件操作失败")
            QMessageBox.warning(self, "导出失败", "无法保存结果，请检查目录的写入权限和剩余磁盘空间。")
            return
        self.status.setText(f"已导出结果、原图与水印选区；各批次 review.html 可逐图复核 → {root}")

    def set_busy(self, busy: bool) -> None:
        busy = busy or self._assist_busy
        self._ui_busy = bool(busy)
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
        self.recall_box.setEnabled(not busy)
        self.style_box.setEnabled(not busy)
        self.pause_btn.setEnabled(busy and not self._assist_busy)
        self.stop_btn.setEnabled(busy and not self._assist_busy)
        self.export_btn.setEnabled(not busy and any(b.results for b in self.batches))
        self._refresh_assisted_controls()

    def _refresh_assisted_controls(self):
        busy = self._ui_busy or self._assist_busy or bool(getattr(self, 'worker', None) and self.worker.isRunning())
        result = self.batch.results.get(self._current) if self.batch else None
        valid = result is not None and not busy
        pending = self._assist_pending is not None
        editing = self._selection_edit is not None
        changed = editing and self._selection_edit.dirty
        self.assist_select_btn.setEnabled(valid and not pending)
        self.assist_confirm_btn.setEnabled(valid and (pending or changed))
        self.assist_cancel_btn.setEnabled(not busy and (pending or editing))
        key = (self.batch.uid, self._current) if self.batch else None
        self.assist_undo_btn.setEnabled(valid and bool(self._assist_undo.get(key)) and not pending and not editing)
        self.brush_tools.setVisible(editing)
        self.brush_tools.setEnabled(not busy)
        self.stroke_undo_btn.setEnabled(not busy and editing and bool(self._selection_edit.undo_stack))
        self.stroke_redo_btn.setEnabled(not busy and editing and bool(self._selection_edit.redo_stack))
        self.param_box.setEnabled(not busy and not editing)
        committed_selection = result is not None and result.mask.selection is not None
        self.reprocess_btn.setText("按已确认选区重新处理" if committed_selection else "处理／重处理当前图片")
        self.auto_redetect_btn.setVisible(committed_selection)
        self.run_btn.setEnabled(not busy and not editing and bool(self.batches))
        self.run_batch_btn.setEnabled(not busy and not editing and bool(self.pages))
        self.export_btn.setEnabled(not busy and not changed and any(b.results for b in self.batches))
        self.export_btn.setToolTip("请先处理或取消未应用的选区编辑" if changed else "导出已处理的结果")
        self.batch_list.setEnabled(not self._assist_busy)
        self.page_list.setEnabled(not self._assist_busy)
        for button in self.view_buttons.values():
            button.setEnabled(not self._assist_busy and not pending)

    def _clear_assisted_preview(self):
        self._selection_edit = None
        self._assist_pending = None
        self.assist_select_btn.blockSignals(True)
        self.assist_select_btn.setChecked(False)
        self.assist_select_btn.blockSignals(False)
        self.canvas.set_selection_mode(False)
        self.canvas.set_brush_mode(False)
        self.canvas.set_selection_overlay()
        self.canvas.clear_search_rectangle()

    def begin_assisted_selection(self, enabled):
        if self._assist_busy:
            return
        if enabled and self.batch and self._current in self.batch.results:
            self._assist_pending = None
            self.canvas.clear_search_rectangle()
            self.canvas.set_selection_mode(False)
            self._selection_edit = SelectionEdit(self.batch.results[self._current])
            self.add_brush_btn.setChecked(True)
            self.set_view_mode("原始图片")
            self.status.setText("在原图上添加或擦除选区；按当前选区处理才更新结果，切图直接丢弃未处理编辑。")
        else:
            self._clear_assisted_preview()
            self.render_canvas()
        self._refresh_assisted_controls()

    def cancel_assisted_selection(self):
        if self._assist_busy:
            return
        self._clear_assisted_preview()
        self.render_canvas()
        self.status.setText("已取消未处理的选区编辑，处理结果未改变")

    def _set_magnifier(self, enabled):
        self.canvas.magnifier_enabled = enabled
        self.canvas.update()

    def _brush_size_changed(self, value):
        self.canvas.brush_diameter = value
        self.canvas.update()

    def set_brush_tool(self, tool):
        self.add_brush_btn.setChecked(tool == 'add')
        self.erase_brush_btn.setChecked(tool == 'erase')
        self.canvas.finish_brush()
        self.canvas.brush_include = tool == 'add'
        self.canvas.update()

    def _begin_brush_stroke(self):
        if self._selection_edit is not None and not self._assist_busy:
            self._selection_edit.begin_stroke()

    def _paint_brush_segment(self, segment):
        if self._selection_edit is not None and not self._assist_busy:
            box = self._selection_edit.paint(*segment)
            if box is not None:
                self.canvas.update_selection_overlay(self._selection_edit.selected, box)
            self.selection_notice.setText("选区已修改，尚未处理；切图丢弃。红色为当前完整选区。")

    def _end_brush_stroke(self):
        if self._selection_edit is not None:
            self._selection_edit.end_stroke()
            self._refresh_assisted_controls()

    def undo_selection_stroke(self):
        if self._selection_edit is not None and not self._assist_busy:
            self._selection_edit.undo()
            self.render_canvas()

    def redo_selection_stroke(self):
        if self._selection_edit is not None and not self._assist_busy:
            self._selection_edit.redo()
            self.render_canvas()

    def find_assisted_strokes(self, rectangle):
        if (self._assist_busy or (self.worker and self.worker.isRunning()) or
                not self.batch or self._current not in self.batch.results):
            return
        self._assist_pending = None
        self._selection_edit = None
        self.canvas.set_brush_mode(False)
        self.canvas.set_selection_overlay()
        result = self.batch.results[self._current]
        self._start_assisted_worker('find', result, rectangle, thr=self.current_thr(),
                                    close_px=self.current_close_px(),
                                    use_ocr=self.ocr_box.isChecked(), style=self.current_style())
        self.status.setText("正在圈选范围内识别水印笔画；尚未修改图片")

    def confirm_assisted_repair(self):
        if self._assist_busy:
            return
        if self._selection_edit is not None and self._selection_edit.dirty:
            result = self.batch.results[self._current]
            request = self._selection_edit.request(self.current_thr())
            self._start_assisted_worker('selection', result, request)
            self.status.setText("正在使用完整选区从原图重新处理；不重新检测水印")
            return
        if not self._assist_pending:
            return
        result = self.batch.results[self._current]
        self._start_assisted_worker('apply', result, self._assist_pending)
        self.status.setText("正在修复已确认的红色笔画")

    def _start_assisted_worker(self, action, result, argument, **options):
        self._assist_context = (self.batch, self._current, result,
                                self.batch.states.get(self._current, "待复核"),
                                self._current in self.batch.dirty)
        self._assist_payload = None
        self._assist_busy = True
        self.canvas.set_selection_mode(False)
        self.canvas.set_brush_mode(False)
        self.assist_worker = AssistWorker(action, result, argument, self, **options)
        self.assist_worker.completed.connect(self._store_assisted_payload)
        self.assist_worker.finished.connect(self._assisted_finished)
        self.set_busy(True)
        self.assist_worker.start()

    def _store_assisted_payload(self, payload):
        self._assist_payload = payload

    def _assisted_finished(self):
        worker = self.assist_worker
        self.assist_worker = None
        self._assist_busy = False
        payload = self._assist_payload
        batch, index, previous, state, dirty = self._assist_context
        self._assist_context = None
        self._assist_payload = None
        self.set_busy(False)
        if not payload:
            self.cancel_assisted_selection()
            self.status.setText("辅助处理未返回结果，原结果已保留")
        elif payload['error']:
            if payload['action'] == 'selection':
                self.render_canvas()
            else:
                self.cancel_assisted_selection()
            self.status.setText(payload['error'])
        elif batch.results.get(index) is not previous:
            self.cancel_assisted_selection()
            self.status.setText("原结果已变化，请重新圈选")
        elif payload['action'] == 'find':
            candidate = payload['value']
            count = int(candidate.layer.binary(candidate.thresholds.mask_alpha_min).astype(bool).sum())
            if count:
                self._assist_pending = candidate
                self.render_canvas()
                self.status.setText(f"圈内找到 {count:,} 个候选笔画像素；请检查红色选区，再确认修复。图片尚未改变。")
            else:
                self.cancel_assisted_selection()
                self.status.setText("圈内未找到有足够证据的水印笔画，请调整圈选范围；图片未改变。")
        else:
            self._assist_undo.setdefault((batch.uid, index), []).append((previous, state, dirty))
            batch.results[index] = payload['value']
            batch.states[index] = ("辅助修复·待复核" if payload['value'].verdict.passed else
                                   "辅助修复·检查未通过")
            batch.dirty.add(index)
            self._clear_assisted_preview()
            self._refresh_assisted_page(index)
            self.status.setText("已按确认选区处理，请复核残留与背景；可撤销本次处理，导出保留自动基线。")
        self._refresh_assisted_controls()
        worker.deleteLater()

    def _refresh_assisted_page(self, index):
        self.results = list(self.batch.results.values())
        self.refresh_batches()
        item = self.page_list.item(index)
        replacement = self._page_item(index)
        item.setText(replacement.text())
        item.setToolTip(replacement.toolTip())
        item.setForeground(replacement.foreground())
        self.show_page(index)

    def undo_assisted_repair(self):
        if self._assist_busy or not self.batch:
            return
        key = (self.batch.uid, self._current)
        history = self._assist_undo.get(key)
        if not history:
            return
        previous, state, dirty = history.pop()
        self.batch.results[self._current] = previous
        self.batch.states[self._current] = state
        if dirty:
            self.batch.dirty.add(self._current)
        else:
            self.batch.dirty.discard(self._current)
        self._clear_assisted_preview()
        self._refresh_assisted_page(self._current)
        self.status.setText("已撤销一次辅助修复，恢复此前结果及选区")

    def start_run(self, only_page: int | None = None, scope: str = "all", *, force_automatic=False) -> None:
        if self._assist_busy or not self.batches or (self.worker and self.worker.isRunning()):
            return
        if self._selection_edit is not None:
            if only_page == self._current:
                self.confirm_assisted_repair()
            else:
                self.status.setText("请先按当前选区处理或取消编辑，再启动自动批量处理。")
            return
        if only_page is not None and self.batch is not None and not force_automatic:
            result = self.batch.results.get(only_page)
            if result is not None and result.mask.selection is not None:
                self._start_assisted_worker('selection', result,
                    SelectionEdit(result).request(self.current_thr()))
                self.status.setText("正在按已确认的完整选区从原图重新处理；保留人工决定")
                return
        self._clear_assisted_preview()
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
                             lama_mask_dilate=vals["lama_mask_dilate"],
                             watermark_recall_priority=batch.recall_priority)
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
            message = payload["message"].replace("PDF", "文档")
            self.status.setText(f"{batch.name} · 第 {index+1} 张 · {message}")
            return
        result = payload["result"]
        self._assist_undo.pop((batch.uid, index), None)
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
                            (" · 请检查文件和本地模型后重试" if payload["error"] else ""))
        self.refresh_batches()
        if batch is self.batch:
            self.results = list(batch.results.values())
            item = self.page_list.item(index)
            replacement = self._page_item(index)
            item.setText(replacement.text())
            item.setToolTip(replacement.toolTip())
            item.setForeground(replacement.foreground())
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
        if self._assist_busy:
            QMessageBox.information(self, "辅助处理运行中", "请等待当前笔画识别或修复完成后再关闭。")
            event.ignore()
            return
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
        if row != self._current:
            self._clear_assisted_preview()
        self._current = row
        self.metrics.setRowCount(0)
        self.conditions.setText("判定：—")
        result = self.batch.results.get(row) if self.batch else None
        if result is not None:
            self._fill_metrics(result)
        elif self.batch and row in self.batch.errors:
            self.conditions.setText("处理失败，请检查输入文件和本地模型后重试。")
        if not (0 <= row < len(self.pages)):
            self.canvas.clear()
            self.preview_info.setText("选择图片后即可预览")
            self.preview_hint.setText("先导入图片，再开始处理")
        self.render_canvas()

        self._refresh_assisted_controls()

    def _fill_metrics(self, result: PageResult) -> None:
        report = result.verdict
        rows = [("处理引擎", {"engine_a_structure": "文档结构处理", "engine_b_raster": "图像修复"}.get(str(report.engine), "其他引擎")),
                ("选区来源", {"structure": "文档结构", "crosspage": "跨页统计", "prior": "图像分析",
                              "model": "模型识别", "human": "人工标记"}.get(str(result.mask.source), "其他来源")),
                ("定位方式", {"repeated_document_overlay": "重复文字与透明度估计",
                              "photo_observed_repeats": "照片重复水印扩展定位",
                              "photo_repeated_colour_inverse": "照片重复水印与透明度反解",
                              "photo_translucent_text": "照片局部透明文字定位",
                              "photo_confirmed_corner_line": "照片角标文字定位",
                              "photo_complete_repeated_rows": "照片重复水印行补全",
                              "photo_confirmed_colour_line": "照片彩色透明文字定位",
                              "repeated_oblique_text": "重复斜向文字",
                              "single_periodic_overlay": "单图点阵与透明度估计",
                              "single_periodic_document": "单图文档点阵与灰度分布估计",
                              "single_repeated_paper": "单图整页重复文字定位",
                              "transparent_pdf_text": "文档透明文字对象移除",
                              "single_chromatic_oblique": "单图色彩残差与大字定位",
                              "single_periodic_texture_text": "周期纹理与透明文字定位",
                              "single_coloured_document_overlay": "单图文档色差与透明度估计",
                              "contrast_dark_oblique": "深色斜向文字定位",
                              "contrast_bright_repeated": "浅色重复文字定位",
                              "contrast_faint_chromatic": "淡色文字定位",
                              "recall_pixel_strokes": "颜色与对比：笔画补全",
                              "recall_colour_refinement": "实验：彩色笔画精细定位",
                              "recall_repeated_strokes": "重复字形：笔画补全",
                              "human_search_strokes": "人工圈选范围内识别笔画",
                              "human_edited_selection": "人工编辑完整水印选区",
                              "corner_id_text": "角标文字定位",
                              "crosspage": "跨页统计"}.get(result.mask.method, "灰度与颜色")),
                ("总耗时", f"{report.elapsed_ms / 1000:.2f} 秒")]
        stages = {"mask_ms": "水印定位耗时", "repair_ms": "图像修复耗时",
                  "assist_mask_ms": "辅助笔画识别耗时", "assist_repair_ms": "辅助修复耗时",
                  "manual_repair_ms": "按选区重新修复耗时"}
        rows += [(stages.get(k, "其他阶段耗时"), f"{v / 1000:.2f} 秒") for k, v in report.stage_ms.items()]
        if result.automatic_result is not None:
            rows.append(("处理模式", "人工辅助 · 自动结果另行保留"))
            rows.append(("已确认辅助次数", str(len(result.assistance_masks))))
        names = {"coverage": "修复选区占比", "fidelity_mean_delta": "选区外平均变化",
                 "fidelity_p99_delta": "选区外变化高位值", "ink_iou": "文字重合度",
                 "residual_basis": "残留评估依据", "residual_coverage": "水印残留率",
                 "mask_iou": "选区重合度", "mask_precision": "选区精确率",
                 "mask_recall": "选区召回率", "veil_s_error": "透明度误差",
                 "veil_t_error": "叠加颜色误差", "psnr": "峰值信噪比",
                 "ssim": "结构相似度", "psnr_in_mask": "选区峰值信噪比",
                 "ink_damage": "文字损伤率", "ink_residue": "文字残留率"}
        for key, value in sorted(report.metrics.items()):
            if key == "residual_basis":
                display = "参考图对照" if value else "图像估计"
            elif not np.isfinite(value):
                display = "未评估" if np.isnan(value) else "无穷大"
            elif key in ("coverage", "residual_coverage", "ink_damage", "ink_residue"):
                display = f"{value:.2%}"
            else:
                display = f"{value:.4f}"
            rows.append((names.get(key, "其他检查指标"), display))
        self.metrics.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(str(k)))
            self.metrics.setItem(i, 1, QTableWidgetItem(str(v)))
        conditions = {"fidelity": "选区外保真", "residual": "水印残留",
                      "recognition": "文字识别", "time_budget": "处理时限"}
        flags = "  ".join("水印残留待复核" if name == "residual" and ok
                          and report.metrics.get("residual_basis") == 0.0 else
                          f"{'✓' if ok else '✕'} {conditions.get(name, '其他检查')}"
                          for name, ok in report.conditions.items())
        if "residual" not in report.conditions:
            flags += "  水印残留未评估"
        failures = {"no_watermark_detected": "未检测到水印", "residual_watermark": "仍有水印残留",
                    "fidelity_breach": "原始内容变化超出阈值", "recognition_regression": "文字识别质量下降",
                    "repair_failed": "图像修复失败", "structure_not_applicable": "不适用文档结构处理",
                    "out_of_time_budget": "处理超过预定时限", "needs_manual_mask": "需要人工标记水印区域"}
        review = "\n待人工复核：自动检查可能漏掉水印。" if report.review_needed else ""
        errors = "；".join(failures.get(str(f), "其他处理异常") for f in report.failures) or "无"
        self.conditions.setText(f"自动检查：{flags or '暂无'}{review}\n异常：{errors}")

    def _page_item(self, index):
        page = self.pages[index]
        state = self.batch.states.get(index, "未处理")
        item = QListWidgetItem(f"{index + 1:02d}  {page.doc_id}\n{state}")
        item.setToolTip(f"{page.doc_id} · 第 {page.page_index + 1} 页 · {state}")
        colors = {"运行失败": "#ae4940", "质量未通过": "#ae4940",
                  "待复核": "#967329", "通过": "#287d55",
                  "辅助修复·待复核": "#967329", "辅助修复·检查未通过": "#ae4940"}
        item.setForeground(QColor(colors.get(state, "#52645a")))
        return item

    def set_view_mode(self, mode: str) -> None:
        if mode not in self.view_buttons:
            return
        self._view_mode = mode
        self.view_buttons[mode].setChecked(True)
        self.render_canvas()

    def render_canvas(self) -> None:
        row = self._current
        if not (0 <= row < len(self.pages)):
            return
        source = self.pages[row].image
        result = self.batch.results.get(row) if self.batch else None
        self._refresh_assisted_controls()
        self.selection_notice.setText("")
        if self._assist_pending is not None and result is not None:
            proposal = self._assist_pending
            selected = proposal.layer.binary(proposal.thresholds.mask_alpha_min) > 0
            img = overlay_mask(result.output.image, selected.astype(np.float32))
            self.canvas.set_images(img, page_key=(self.batch.uid, row), caption="候选笔画·尚未修复")
            self.preview_hint.setText("红色为将要修复的候选笔画，黄框仅为搜索范围。请确认没有误选背景；确认前图片不变。")
            return
        mode = self._view_mode
        comparing = result is not None and mode == "拖拽对比"
        self.preview_info.setText(f"第 {row + 1} / {len(self.pages)} 张 · {source.shape[1]} × {source.shape[0]} 像素")
        if result is None:
            img, caption = source, "原图"
            self.preview_hint.setText("当前图片尚无处理结果，处理完成后可拖拽对比")
        elif mode == "原始图片" or comparing:
            img, caption = source, "原图"
        elif mode == "水印选区":
            if self._selection_edit is not None or result.mask.selection is not None:
                img, caption = source, "完整水印选区"
            else:
                img, caption = overlay_mask(source, result.mask.alpha), "水印选区"
        else:
            img, caption = result.output.image, "处理结果"
        if result is not None:
            self.preview_hint.setText("左右拖动分隔线查看变化 · 双击居中 · 方向键微调" if comparing else
                                      "红色区域为自动检测到的水印选区" if mode == "水印选区" else
                                      "可切换至「拖拽对比」查看处理前后的变化")
        self.canvas.set_images(img, result.output.image if comparing else None,
                               page_key=(self.batch.uid, row), caption=caption)
        editable_view = mode in ("原始图片", "水印选区")
        editing = self._selection_edit is not None
        self.canvas.set_brush_mode(editing and editable_view and not self._assist_busy,
                                   include=self.add_brush_btn.isChecked())
        if editing:
            self.canvas.set_selection_overlay(self._selection_edit.selected if editable_view else None)
            self.selection_notice.setText(
                "选区已修改，尚未处理；切图丢弃。红色为当前完整选区。" if self._selection_edit.dirty else
                "编辑完整选区：添加或擦除，最后一笔优先；点击处理才更新图片。")
            self.preview_hint.setText("滚轮缩放 · 空格＋拖动平移 · B 添加 / E 擦除 · 按住 H 隐藏选区 · ⌘/Ctrl＋Z 撤销" if editable_view else
                                      "正在查看已有处理结果；切回原始图片或水印选区继续描涂，编辑仍保留。")
        elif result is not None and result.mask.selection is not None and mode == "水印选区":
            self.canvas.set_selection_overlay(result.mask.selection)
            self.preview_hint.setText("红色为该处理结果使用的完整选区，包含人工添加与擦除的修改。")
        else:
            self.canvas.set_selection_overlay()


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
          f"elapsed={report.elapsed_ms}ms canvas={win.canvas.before.width()}x{win.canvas.before.height()}")
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
        win.set_view_mode(mode)
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
