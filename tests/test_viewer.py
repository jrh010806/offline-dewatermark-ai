import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import numpy as np
from PySide6.QtCore import QPoint, Qt, QCoreApplication
from PySide6.QtTest import QTest

from wm.app import MainWindow, make_app, VIEW_MODES, STYLE_LABELS
from wm.batches import Batch
from wm.contract import PageImage, PageResult, MaskLayer, VerdictReport, EngineKind, FailureCode
from wm.viewer import ImageViewer


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def setUp(self):
        self.viewer = ImageViewer()
        self.viewer.resize(600, 500)
        self.before = np.full((300, 200, 3), (240, 30, 40), np.uint8)
        self.after = np.full((300, 200, 3), (20, 60, 230), np.uint8)
        self.viewer.set_images(self.before, self.after, page_key="one")
        self.viewer.show()
        self.app.processEvents()

    def tearDown(self):
        self.viewer.close()

    def pixel(self, fraction):
        rect = self.viewer.image_rect()
        shot = self.viewer.grab().toImage()
        dpr = shot.devicePixelRatio()
        return shot.pixelColor(int((rect.x() + rect.width() * fraction) * dpr),
                               int((rect.y() + rect.height() * 0.2) * dpr)).getRgb()[:3]

    def test_drag_changes_pixels_and_clamps_at_image_edges(self):
        self.assertEqual(self.pixel(0.2), (240, 30, 40))
        self.assertEqual(self.pixel(0.8), (20, 60, 230))
        rect = self.viewer.image_rect()
        QTest.mousePress(self.viewer, Qt.LeftButton, pos=rect.center().toPoint())
        QTest.mouseMove(self.viewer, QPoint(int(rect.x() + rect.width() * 0.8), int(rect.center().y())))
        self.assertAlmostEqual(self.viewer.fraction, 0.8, delta=0.01)
        self.assertEqual(self.pixel(0.7), (240, 30, 40))
        QTest.mouseMove(self.viewer, QPoint(0, int(rect.center().y())))
        self.assertEqual(self.viewer.fraction, 0)
        self.assertEqual(self.pixel(0.2), (20, 60, 230))
        QTest.mouseRelease(self.viewer, Qt.LeftButton, pos=QPoint(self.viewer.width() - 1, 200))
        self.assertEqual(self.viewer.fraction, 1)
        self.assertEqual(self.pixel(0.8), (240, 30, 40))

    def test_resize_keeps_alignment_fraction_and_aspect_ratio(self):
        self.viewer.set_fraction(0.3)
        for size in [(900, 400), (320, 700)]:
            self.viewer.resize(*size)
            self.app.processEvents()
            rect = self.viewer.image_rect()
            self.assertAlmostEqual(rect.width() / rect.height(), 2 / 3)
            self.assertEqual(self.pixel(0.2), (240, 30, 40))
            self.assertEqual(self.pixel(0.8), (20, 60, 230))
            self.assertEqual(self.viewer.fraction, 0.3)

    def test_keyboard_double_click_and_page_reset(self):
        QTest.keyClick(self.viewer, Qt.Key_Right)
        self.assertAlmostEqual(self.viewer.fraction, 0.52)
        QTest.keyClick(self.viewer, Qt.Key_Home)
        self.assertEqual(self.viewer.fraction, 0)
        QTest.keyClick(self.viewer, Qt.Key_End)
        self.assertEqual(self.viewer.fraction, 1)
        QTest.mouseDClick(self.viewer, Qt.LeftButton, pos=QPoint(300, 200))
        self.assertEqual(self.viewer.fraction, 0.5)
        self.viewer.set_fraction(0.7)
        self.viewer.set_images(self.before, self.after, page_key="one")
        self.assertEqual(self.viewer.fraction, 0.7)
        self.viewer.set_images(self.before, self.after, page_key="two")
        self.assertEqual(self.viewer.fraction, 0.5)

    def test_without_result_does_not_offer_fake_comparison(self):
        self.viewer.set_images(self.before, page_key="pending")
        QTest.mouseClick(self.viewer, Qt.LeftButton, pos=QPoint(20, 200))
        self.assertFalse(self.viewer.comparing)
        self.assertEqual(self.viewer.fraction, 0.5)
        self.assertEqual(self.pixel(0.8), (240, 30, 40))
        self.viewer.clear()
        self.assertTrue(self.viewer.before.isNull())


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def setUp(self):
        self.win = MainWindow()
        source = PageImage(np.full((100, 80, 3), 240, np.uint8), 200, "测试图片")
        output = PageImage(np.full((100, 80, 3), 200, np.uint8), 200, "测试图片")
        report = VerdictReport(source.key, EngineKind.RASTER, [],
                               {"fidelity": True, "residual": True, "time_budget": True},
                               {"coverage": 0.1, "ink_iou": float("nan")},
                               [FailureCode.NO_WATERMARK_DETECTED], 1800,
                               {"mask_ms": 1200, "repair_ms": 600})
        batch = Batch("测试批次", [source, source])
        batch.results[0] = PageResult(source, output, MaskLayer(np.zeros((100, 80))), report)
        batch.style = "tiled_text"
        self.win.batches = [batch]
        self.win.refresh_batches()
        self.win.batch_list.setCurrentRow(0)

    def tearDown(self):
        self.win.close()

    def test_view_modes_and_pending_page(self):
        w = self.win
        self.assertTrue(w.canvas.comparing)
        w.canvas.set_fraction(0.25)
        for mode in VIEW_MODES[1:]:
            w.view_buttons[mode].click()
            self.assertFalse(w.canvas.comparing)
        w.view_buttons["拖拽对比"].click()
        self.assertEqual(w.canvas.fraction, 0.25)
        w.page_list.setCurrentRow(1)
        self.assertFalse(w.canvas.comparing)
        self.assertEqual(w.canvas.fraction, 0.5)
        w.page_list.setCurrentRow(0)
        self.assertTrue(w.canvas.comparing)
        w.delete_batch()
        self.assertTrue(w.canvas.before.isNull())

    def test_chinese_metrics_styles_and_standard_dialogs(self):
        w = self.win
        self.assertEqual(w.style_box.currentText(), "平铺文字")
        self.assertEqual(w.current_style(), "tiled_text")
        for code, label in STYLE_LABELS.items():
            w.style_box.setCurrentText(label)
            self.assertEqual(w.current_style(), code)
        texts = [w.metrics.item(row, col).text() for row in range(w.metrics.rowCount()) for col in range(2)]
        self.assertIn("未评估", texts)
        self.assertIn("图像修复", texts)
        self.assertFalse(any(any(c.isascii() and c.isalpha() for c in text) for text in texts))
        self.assertIn("未检测到水印", w.conditions.text())
        self.assertNotIn("fidelity", w.conditions.text())
        self.assertNotEqual(QCoreApplication.translate("QPlatformTheme", "Cancel"), "Cancel")

    def test_photo_paths_are_named_and_unreferenced_residue_is_not_a_pass(self):
        result = self.win.batch.results[0]
        result.verdict.metrics['residual_basis'] = 0.0
        for method, label in [('photo_observed_repeats', '照片重复水印扩展定位'),
                              ('photo_translucent_text', '照片局部透明文字定位')]:
            result.mask.method = method
            self.win._fill_metrics(result)
            texts = [self.win.metrics.item(row, col).text()
                     for row in range(self.win.metrics.rowCount()) for col in range(2)]
            self.assertIn(label, texts)
            self.assertIn('修复选区占比', texts)
            self.assertIn('水印残留待复核', self.win.conditions.text())
            self.assertNotIn('✓ 水印残留', self.win.conditions.text())
        result.verdict.metrics['residual_basis'] = 1.0
        self.win._fill_metrics(result)
        self.assertIn('✓ 水印残留', self.win.conditions.text())
        result.verdict.metrics['residual_basis'] = 0.0
        result.verdict.conditions['residual'] = False
        self.win._fill_metrics(result)
        self.assertIn('✕ 水印残留', self.win.conditions.text())


if __name__ == "__main__":
    unittest.main()
