"""角标专用通道与无正文区域识别条件的回归测试。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from wm import corner_id, verdict
from wm.contract import EngineKind, MaskLayer, Thresholds


def _page(with_id: bool):
    img = np.full((400, 300, 3), 255, np.uint8)
    if with_id:
        import cv2
        cv2.putText(img, "ZG-053-8F3A", (170, 370), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def _obs():
    # 角区裁剪图（135x80）放大 2 倍后的坐标，对应原图 (170,370) 处的 ID 文字。
    box = np.array([[10, 96], [230, 96], [230, 140], [10, 140]], np.float32)
    return [(box, "ZG-053-8F3A", .99)]


class CornerIdTests(unittest.TestCase):
    def test_detects_and_cleans_corner_identifier(self):
        with patch('wm.corner_id.ocr.read_lines', return_value=_obs()):
            mask = corner_id.estimate(_page(True))
        self.assertIsNotNone(mask)
        self.assertEqual(mask.method, 'corner_id_text')
        self.assertEqual(mask.repair_mode, 'corner_cleanup')
        self.assertGreater(float(mask.alpha.sum()), 0)

    def test_page_without_id_text_is_declined(self):
        with patch('wm.corner_id.ocr.read_lines', return_value=[]):
            self.assertIsNone(corner_id.estimate(_page(False)))

    def test_page_number_is_not_an_identifier(self):
        self.assertFalse(corner_id._id_like("3"))
        self.assertFalse(corner_id._id_like("第3页"))
        self.assertTrue(corner_id._id_like("ZG-053-8F3A"))
        self.assertTrue(corner_id._id_like("CONF-2026"))

    def test_document_group_and_page_labels_are_not_identifiers(self):
        self.assertFalse(corner_id._id_like("FICTITIOUS TRAINING MATERIAL | GROUP 025 | PAGE 1"))
        self.assertFalse(corner_id._id_like("INTERNAL SAMPLE 025/01"))

    def test_short_identifier_inside_table_is_body_content(self):
        import cv2
        image = np.full((400, 500), 255, np.uint8)
        for y in (40, 80, 120):
            cv2.line(image, (20, y), (450, y), 190, 1)
        for x in (20, 230, 450):
            cv2.line(image, (x, 40), (x, 120), 190, 1)
        box = np.array([[30, 50], [95, 50], [95, 70], [30, 70]], np.float32)
        self.assertTrue(corner_id._inside_table(image, box, 255))
        image[:, 230] = 255
        self.assertFalse(corner_id._inside_table(image, box, 255))


class RecognitionVacuousTests(unittest.TestCase):
    def test_recognition_passes_when_region_has_no_body_text(self):
        clean = np.full((40, 40, 3), 255, np.uint8)
        mask = MaskLayer(alpha=np.ones((40, 40), np.float32),
                         veil_strength=np.zeros(3, np.float32),
                         veil_offset=np.zeros(3, np.float32))
        report = verdict.build_verdict(source=clean, output=clean, mask=mask,
                                       engine=EngineKind.RASTER, repairs=[], elapsed_ms=1,
                                       stage_ms={}, thr=Thresholds(),
                                       clean_reference=clean,
                                       alpha_reference=np.ones((40, 40), np.float32))
        self.assertEqual(report.metrics["ink_iou"], 1.0)
        self.assertTrue(report.conditions["recognition"])


if __name__ == "__main__":
    unittest.main()
