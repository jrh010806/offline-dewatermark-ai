import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import paper_grid, repair
from wm.contract import Thresholds


class PaperGridTests(unittest.TestCase):
    @staticmethod
    def fixture(period):
        h, w = 500, 550
        clean = np.full((h, w, 3), 245, np.uint8)
        for y in range(45, 480, 27):
            cv2.putText(clean, 'Body text 2037', (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (10, 10, 10), 2)
        cv2.circle(clean, (450, 420), 45, (180, 25, 30), 3)
        yy, xx = np.indices((h, w))
        dx = (xx-5+period/2) % period-period/2
        dy = (yy-8+period/2) % period-period/2
        alpha = np.clip(2.4-np.sqrt(dx*dx+dy*dy), 0, 1)*.5
        image = np.uint8(clean*(1-alpha[:, :, None])+alpha[:, :, None]*np.array([125, 125, 135]))
        return clean, image, alpha

    def test_unseen_pitches_preserve_print_and_red_seal_without_neural_inference(self):
        for pitch in (14, 22):
            with self.subTest(pitch=pitch):
                clean, source, alpha = self.fixture(pitch)
                mask = paper_grid.estimate(source)
                self.assertIsNotNone(mask)
                self.assertEqual(mask.lattice, (pitch, pitch))
                with patch('wm.lama._load_session', side_effect=AssertionError('paper dots need no generative model')):
                    output, _ = repair.repair(source, mask, Thresholds())
                region = alpha > .05
                before = abs(source.astype(float)-clean)[region].mean()
                after = abs(output.astype(float)-clean)[region].mean()
                self.assertLess(after, before*.1)
                np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
                printed = clean.mean(2) < 30
                self.assertLess(np.mean(output[printed].mean(1) > 80), .01)
                seal = clean[:, :, 0] > clean[:, :, 1].astype(float)+50
                self.assertLess(abs(output.astype(float)-clean)[seal].mean(), 8)

    def test_queue_avoids_ocr_before_source_only_periodic_detection(self):
        from wm.batches import Batch, BatchWorker
        from wm.contract import PageImage
        _, source, _ = self.fixture(14)
        batch = Batch('dots', [PageImage(source, 200)])
        worker = BatchWorker([(batch, [0], Thresholds(), 11, True)])
        events = []; worker.update.connect(events.append)
        with patch('wm.ocr.text_zones', side_effect=AssertionError('unnecessary OCR')):
            worker.run()
        finished = [e for e in events if e['kind']=='page']
        self.assertEqual(len(finished), 1)
        self.assertIsNone(finished[0]['error'])
        self.assertEqual(finished[0]['result'].mask.method, 'single_periodic_document')

    def test_unmarked_document_and_repeated_solid_rules_are_rejected(self):
        clean, _, _ = self.fixture(14)
        self.assertIsNone(paper_grid.estimate(clean))
        rules = np.full_like(clean, 245)
        rules[::18] = 0
        rules[:, ::18] = 0
        self.assertIsNone(paper_grid.estimate(rules))


if __name__ == '__main__':
    unittest.main()
