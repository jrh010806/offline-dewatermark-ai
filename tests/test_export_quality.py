import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wm import pipeline
from wm.contract import PageImage,PageResult,MaskLayer,VerdictReport,EngineKind


class ExportQualityTests(unittest.TestCase):
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
        report=VerdictReport(source.key,EngineKind.RASTER,[],{'recognition':True},
                            {'ink_iou':1.},[],10,has_ground_truth=True)
        result=PageResult(source,source,MaskLayer(np.zeros((20,30),np.float32)),report)
        with tempfile.TemporaryDirectory() as folder:
            pipeline.write_results([result],folder)
            data=json.loads((Path(folder)/'manifest.json').read_text())
        self.assertEqual(data['summary']['success_rate'],1.)
        self.assertEqual(data['summary']['quality_evaluated'],1)

if __name__=='__main__':unittest.main()
