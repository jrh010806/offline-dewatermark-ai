import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.run_mask_contract_trace import Trace
from wm import lama, masks, repair
from wm.contract import MaskLayer, Thresholds


class FakeSession:
    def run(self, names, feeds):
        # A deterministic model stub makes geometry and tracing side effects
        # observable without pretending to test image restoration quality.
        return [np.full_like(feeds['image'], 73)]


class MaskContractTraceTests(unittest.TestCase):
    def check_capture(self, shape, dilation):
        source = np.full((*shape, 3), 180, np.uint8)
        selected = np.zeros(shape, bool)
        selected[20:30, 35:49] = True
        selected[-30:-18, -49:-35] = True
        selected[22:27, 39:45] = False
        thr = Thresholds(lama_mask_dilate=dilation, lama_context_margin=8)
        base = MaskLayer(selected.astype(np.float32), footprint=selected,
                         repair_mode='inpaint', method='test_observed_strokes')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / '源图.png'
            Image.fromarray(source).save(path)
            task = dict(inputs=[dict(id='test', input=str(path), input_sha256='test')],
                        settings=dict(thresholds=asdict(thr)))
            with patch.object(lama, '_load_session', return_value=FakeSession()):
                expected, _ = repair.repair(source, base, thr)
            trace = Trace(task, folder)
            with ExitStack() as stack:
                stack.enter_context(patch.object(masks, '_estimate_single_page', return_value=base))
                stack.enter_context(patch.object(lama, '_load_session', return_value=FakeSession()))
                trace.install(stack)
                mask = masks.estimate_single_page(source, thr)
                mask = masks.with_prior(mask, None, thr)
                actual, _ = repair.repair(source, mask, thr)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(base.alpha, selected.astype(np.float32))
            row = json.loads(next(folder.glob('逐阶段_*/逐阶段记录.json')).read_text())
            call = row['lama_calls'][0]
            model_mask = np.asarray(Image.open(Path(row['directory']) /
                '06_LaMa调用0_模型实际输入选区.png')) > 0
            self.assertEqual(call['lost_before_to_model'], 0)
            self.assertEqual(call['crop_mask_pixels'], int(model_mask.sum()))
            self.assertGreater(len(call['tiles']), 0)
            if dilation == 0:
                np.testing.assert_array_equal(model_mask, selected)
                self.assertEqual(row['outside_selected_changed'], 0)
            else:
                self.assertGreater(call['added_before_to_model'], 0)
            np.testing.assert_array_equal(actual[~model_mask], source[~model_mask])

    def test_small_padded_crop_preserves_holes_and_output(self):
        self.check_capture((100, 140), 0)

    def test_tiled_crop_maps_real_model_masks_to_original_coordinates(self):
        self.check_capture((570, 890), 0)

    def test_explicit_dilation_is_reported_separately(self):
        self.check_capture((100, 140), 2)


if __name__ == '__main__':
    unittest.main()
