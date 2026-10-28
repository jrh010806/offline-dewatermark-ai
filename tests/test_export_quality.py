import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from wm import pipeline
from wm.export_review import overlay_mask
from wm.contract import PageImage,PageResult,MaskLayer,VerdictReport,EngineKind


class ExportQualityTests(unittest.TestCase):
    def test_export_preserves_selection_and_uses_frozen_processing_threshold(self):
        source = PageImage(np.full((2, 3, 3), 180, np.uint8), 72, '样例')
        alpha = np.array([[0., .0799, .08], [.1999, .2, 1.]], np.float32)
        footprint = alpha > .1
        report = VerdictReport(source.key, EngineKind.RASTER, [], {}, {}, [], 10)
        mask = MaskLayer(alpha, footprint=footprint, text_keepout=~footprint)
        result = PageResult(source, source, mask, report,
                            processing={'thresholds': {'mask_alpha_min': .2}})
        with tempfile.TemporaryDirectory() as folder:
            root = pipeline.write_results([result], folder)
            manifest = json.loads((root/'manifest.json').read_text())
            page = manifest['pages'][0]
            files = page['files']
            for path in files.values():
                self.assertTrue((root/path).is_file())
            np.testing.assert_array_equal(np.array(Image.open(root/files['source'])), source.image)
            np.testing.assert_array_equal(np.array(Image.open(root/files['mask'])),
                                          np.array([[0, 0, 0], [0, 255, 255]], np.uint8))
            np.testing.assert_array_equal(np.array(Image.open(root/files['selection_overlay'])),
                                          overlay_mask(source.image, alpha))
            with np.load(root/files['mask_data'], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['alpha'], alpha)
                np.testing.assert_array_equal(raw['footprint'], footprint)
                np.testing.assert_array_equal(raw['text_keepout'], ~footprint)
            self.assertEqual(page['selection']['selected_pixels'], 2)
            self.assertEqual(page['selection']['threshold'], .2)
            self.assertAlmostEqual(page['coverage'], 1/3, places=5)
            self.assertIn('水印选区', (root/'水印选区与处理结果复核.html').read_text())
            self.assertNotIn('watermark', files)

    def test_duplicate_and_unsafe_names_preserve_every_page_and_escape_html(self):
        report = VerdictReport('sample', EngineKind.RASTER, [], {}, {}, [], 10)
        results = []
        for value in (30, 200):
            source = PageImage(np.full((2, 3, 3), value, np.uint8), 72, '../../<script>')
            results.append(PageResult(source, source, MaskLayer(np.zeros((2, 3))), report))
        with tempfile.TemporaryDirectory() as folder:
            root = pipeline.write_results(iter(results), folder)
            pages = json.loads((root/'manifest.json').read_text())['pages']
            self.assertNotEqual(pages[0]['files']['source'], pages[1]['files']['source'])
            for page, value in zip(pages, (30, 200)):
                for path in page['files'].values():
                    self.assertEqual((root/path).parent, root)
                self.assertTrue(np.all(np.array(Image.open(root/page['files']['source'])) == value))
            html = (root/'水印选区与处理结果复核.html').read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)

    def test_unreviewed_pages_do_not_become_one_hundred_percent_success(self):
        source=PageImage(np.full((20,30,3),255,np.uint8),72,'sample')
        report=VerdictReport(source.key,EngineKind.RASTER,[],{'fidelity':True},
                            {'ink_iou':float('nan')},[],10)
        result=PageResult(source,source,MaskLayer(np.zeros((20,30),np.float32)),report,
                          processing={'mask_close_px':11})
        with tempfile.TemporaryDirectory() as folder:
            pipeline.write_results([result],folder)
            data=json.loads((Path(folder)/'manifest.json').read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        self.assertIsNone(data['summary']['success_rate'])
        self.assertEqual(data['summary']['passed'],0)
        self.assertEqual(data['summary']['checks_passed'],1)
        self.assertEqual(data['summary']['needs_review'],1)
        self.assertIsNone(data['pages'][0]['verdict']['metrics']['ink_iou'])
        self.assertEqual(data['pages'][0]['processing']['mask_close_px'],11)

    def test_reference_evaluated_pages_report_measured_success(self):
        source=PageImage(np.full((20,30,3),255,np.uint8),72,'sample')
        report=VerdictReport(source.key,EngineKind.RASTER,[],{'recognition':True,'residual':True},
                            {'ink_iou':1.},[],10,has_ground_truth=True)
        result=PageResult(source,source,MaskLayer(np.zeros((20,30),np.float32)),report)
        with tempfile.TemporaryDirectory() as folder:
            pipeline.write_results([result],folder)
            data=json.loads((Path(folder)/'manifest.json').read_text())
        self.assertEqual(data['summary']['success_rate'],1.)
        self.assertEqual(data['summary']['quality_evaluated'],1)

if __name__=='__main__':unittest.main()
