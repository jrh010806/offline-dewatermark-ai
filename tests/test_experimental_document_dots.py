import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.experimental_document_dots import METHOD, enabled, estimate
from wm.contract import Thresholds


def fixture(px=43, py=37, shift=17):
    clean = np.full((730, 930, 3), 247, np.uint8)
    cv2.rectangle(clean, (350, 250), (640, 560), (45, 85, 125), -1)
    for y in range(100, 650, 65):
        cv2.putText(clean, 'Document 2397 / AB', (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (18, 18, 18), 2)
    support = np.zeros(clean.shape[:2], np.uint8)
    for row, y in enumerate(range(17, clean.shape[0], py)):
        for x in range(13+(row % 2)*shift, clean.shape[1], px):
            cv2.circle(support, (x, y), 2, 1, -1)
    alpha = support.astype(float)*.25
    source = np.uint8(clean*(1-alpha[..., None])
                      + np.array([130, 105, 125])*alpha[..., None])
    return clean, source, support > 0


class ExperimentalDocumentDotTests(unittest.TestCase):
    def test_rectangular_staggered_lattices_use_measured_shapes(self):
        for px, py, shift in ((43, 37, 17), (31, 47, 12), (59, 41, 27)):
            with self.subTest(px=px, py=py, shift=shift):
                clean, source, truth = fixture(px, py, shift)
                trace = []
                mask = estimate(source, trace)
                self.assertIsNotNone(mask, trace)
                selected = mask.binary() > 0
                self.assertEqual(int((selected & ~truth).sum()), 0)
                self.assertGreater(float((selected & truth).sum()/truth.sum()), .7)
                self.assertEqual(trace[0]['pitch_x'], px)
                self.assertEqual(trace[0]['row_pitch'], py)
                from wm import repair
                with enabled(), patch('wm.lama._load_session', side_effect=AssertionError('no LaMa')):
                    output, _ = repair.repair(source, mask, Thresholds())
                np.testing.assert_array_equal(output[~selected], source[~selected])
                self.assertLess(np.abs(output.astype(float)-clean)[selected].mean(), 1.2)

    def test_blank_body_text_table_and_sparse_bullets_do_not_make_dot_selection(self):
        clean, _, _ = fixture()
        blank = np.full_like(clean, 255)
        grid = blank.copy()
        grid[::31] = 90
        grid[:, ::43] = 90
        bullets = clean.copy()
        for y in range(50, 700, 40):
            cv2.circle(bullets, (25, y), 2, (160, 160, 160), -1)
        for source in (blank, clean, grid, bullets, np.full_like(clean, 35)):
            self.assertIsNone(estimate(source))

    def test_regular_square_dots_keep_the_original_path(self):
        _, source, _ = fixture(29, 29, 0)
        self.assertIsNone(estimate(source))

    def test_damaged_text_intersections_are_left_unmodified(self):
        clean, source, _ = fixture()
        mask = estimate(source)
        self.assertIsNotNone(mask)
        selected = mask.binary() > 0
        printed = clean.mean(2) < 30
        self.assertEqual(int((selected & printed).sum()), 0)

    def test_adapter_prefers_existing_model_and_uninstalls_after_exception(self):
        from wm import paper_grid, masks, recall, repair
        originals = (paper_grid.estimate, masks.with_prior, recall.expand, repair.repair)
        sentinel = object()
        with patch('wm.paper_grid.estimate', return_value=sentinel), enabled():
            self.assertIs(paper_grid.estimate(np.zeros((30, 30, 3), np.uint8)), sentinel)
        try:
            with enabled():
                raise RuntimeError('cancelled research process')
        except RuntimeError:
            pass
        self.assertEqual(originals, (paper_grid.estimate, masks.with_prior, recall.expand, repair.repair))

    def test_batchworker_exercises_candidate_and_preserves_outside_selection(self):
        from wm.batches import Batch, BatchWorker
        from wm.contract import PageImage
        _, source, _ = fixture()
        batch = Batch('research fixture', [PageImage(source, 200)])
        events = []
        worker = BatchWorker([(batch, [0], Thresholds(watermark_recall_priority=True), 11, True)])
        worker.update.connect(events.append)
        with enabled(), patch('wm.lama._load_session', side_effect=AssertionError('no LaMa')):
            worker.run()
        result = [e for e in events if e['kind'] == 'page'][0]
        self.assertIsNone(result['error'])
        self.assertEqual(result['result'].mask.method, METHOD)
        outside = result['result'].mask.binary() == 0
        np.testing.assert_array_equal(result['result'].output.image[outside], source[outside])


if __name__ == '__main__':
    unittest.main()
