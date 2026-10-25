"""样式路由、提示词翻译与重绘硬约束的回归测试。"""
from __future__ import annotations

import unittest

import numpy as np

from wm import masks
from wm.contract import MaskLayer, Thresholds
from wm.lama import repair_with_lama
from wm.styles import STYLE_CHANNELS, parse_style_prompt, translate_prompt


class StylePromptTests(unittest.TestCase):
    def test_codes_and_bilingual_prompts(self):
        self.assertEqual(parse_style_prompt("tiled_text"), "tiled_text")
        self.assertEqual(parse_style_prompt("平铺文字"), "tiled_text")
        self.assertEqual(parse_style_prompt("斜向平铺文字水印"), "tiled_text")
        self.assertEqual(parse_style_prompt("Tiled Text Watermark"), "tiled_text")
        self.assertEqual(parse_style_prompt("点阵"), "dot_grid")
        self.assertEqual(parse_style_prompt("dot grid"), "dot_grid")
        self.assertEqual(parse_style_prompt("深色重字母"), "heavy_dark")
        self.assertEqual(parse_style_prompt("heavy dark"), "heavy_dark")
        self.assertEqual(parse_style_prompt("角标"), "corner_id")
        self.assertIsNone(parse_style_prompt("自动识别"))
        self.assertIsNone(parse_style_prompt(""))

    def test_translate_prompt(self):
        r = translate_prompt("平铺文字")
        self.assertEqual(r["style"], "tiled_text")
        self.assertIn("tiled", r["english"])
        self.assertIn("平铺", r["chinese"])

    def test_channels_cover_all_styles(self):
        for style in ("dot_grid", "tiled_text", "big_center", "heavy_dark", "corner_id"):
            self.assertIn(style, STYLE_CHANNELS)


class LockedRoutingTests(unittest.TestCase):
    def test_locked_style_without_evidence_returns_empty(self):
        img = np.full((200, 200, 3), 255, np.uint8)
        mask = masks.estimate_single_page(img, Thresholds(), use_ocr=False, style="dot_grid")
        self.assertEqual(float(mask.alpha.max()), 0.0)

    def test_corner_id_keeps_generic_path(self):
        img = np.full((200, 200, 3), 255, np.uint8)
        mask = masks.estimate_single_page(img, Thresholds(), use_ocr=False, style="corner_id")
        self.assertIsNotNone(mask)


class RepaintGuardTests(unittest.TestCase):
    def test_huge_footprint_never_repaints_whole_page(self):
        img = np.full((64, 64, 3), 200, np.uint8)
        mask = MaskLayer(alpha=np.full((64, 64), 0.5, np.float32),
                         veil_strength=np.full(3, 0.5, np.float32),
                         veil_offset=np.full(3, 100, np.float32),
                         confidence=0.9,
                         footprint=np.ones((64, 64), bool))
        out, used = repair_with_lama(img, mask, Thresholds())
        self.assertNotIn("lama", [str(u) for u in used])
        self.assertLessEqual(mask.confidence, 0.05)


class PriorGuardTests(unittest.TestCase):
    def test_paper_repeat_method_never_overridden_by_prior(self):
        single = MaskLayer(alpha=np.ones((4, 4), np.float32), method="paper_repeat_document_overlay")
        prior = MaskLayer(alpha=np.zeros((4, 4), np.float32))
        self.assertIs(masks.with_prior(single, prior, Thresholds()), single)


if __name__ == "__main__":
    unittest.main()
