import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf as fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject, NumberObject, TextStringObject

from wm import documents, pipeline, structure
from wm.batches import Batch, BatchWorker
from wm.contract import Thresholds
from wm.pdf_text import strip_translucent_text


def fixture(angle=30, opacity=.25, repeats=1):
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    for y in range(40, 360, 20):
        page.insert_text((20, y), 'Normal body text 123456789, table content.', fontsize=11)
    clean = doc.tobytes()
    for i in range(repeats):
        point = fitz.Point(50, 160+i*60)
        page.insert_text(point, 'Overlay (test)', fontsize=32, fill_opacity=opacity,
                         morph=(point, fitz.Matrix(1, 1).prerotate(angle)))
    source = doc.tobytes()
    doc.close()
    return clean, source


def pixels(data):
    with fitz.open(stream=data, filetype='pdf') as doc:
        pix = doc[0].get_pixmap(colorspace=fitz.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)


class PDFTextTests(unittest.TestCase):
    def test_operator_removal_preserves_underlying_body_exactly(self):
        clean, source = fixture()
        output, counts = strip_translucent_text(source)
        self.assertEqual(counts, (1,))
        np.testing.assert_array_equal(pixels(output), pixels(clean))
        with fitz.open(stream=output, filetype='pdf') as doc:
            self.assertIn('Normal body', doc[0].get_text())
            self.assertNotIn('Overlay', doc[0].get_text())

    def test_repeated_horizontal_overlay_and_legitimate_text_guards(self):
        clean, source = fixture(angle=0, repeats=3)
        output, counts = strip_translucent_text(source)
        self.assertEqual(counts, (3,))
        np.testing.assert_array_equal(pixels(output), pixels(clean))
        for angle, alpha in [(0, .25), (30, 1.), (0, 1.)]:
            _, source = fixture(angle=angle, opacity=alpha)
            output, counts = strip_translucent_text(source)
            self.assertEqual(counts, (0,))
            self.assertEqual(output, source)
        output, counts = strip_translucent_text(clean)
        self.assertEqual(counts, (0,))
        self.assertEqual(output, clean)

    def test_mixed_body_and_overlay_in_one_text_block_is_untouched(self):
        _, source = fixture()
        reader = PdfReader(BytesIO(source))
        page = reader.pages[0]
        stream = ContentStream(page.get_contents(), reader)
        ops = stream.operations
        target = max(i for i, (_, op) in enumerate(ops) if op in (b'Tj', b'TJ'))
        # Put a smaller, legitimate show in the same BT/ET block.
        ops[target+1:target+1] = [([NameObject('/helv'), NumberObject(11)], b'Tf'),
                                ([TextStringObject('BODY')], b'Tj')]
        stream.operations = ops
        page[NameObject('/Contents')] = stream
        writer = PdfWriter(); writer.add_page(page)
        buf = BytesIO(); writer.write(buf)
        source = buf.getvalue()
        output, counts = strip_translucent_text(source)
        self.assertEqual(counts, (0,))
        self.assertEqual(output, source)

    def test_batch_uses_original_pdf_page_and_updates_each_result_without_raster(self):
        clean, source = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.pdf'; path.write_bytes(source)
            pages = documents.load_many([path, path], dpi=72)
            self.assertEqual([p.source_page_index for p in pages], [0, 0])
            batch = Batch('pdf', pages)
            worker = BatchWorker([(batch, [0, 1], Thresholds(), 11, False)])
            events = []; worker.update.connect(events.append)
            with patch('wm.batches.masks.estimate_single_page', side_effect=AssertionError('raster called')):
                worker.run()
            finished = [e for e in events if e['kind']=='page']
            self.assertEqual([e['done'] for e in finished], [1, 2])
            for event in finished:
                self.assertIsNone(event['error'])
                np.testing.assert_array_equal(event['result'].output.image, pixels(clean))
                self.assertEqual(event['result'].mask.method, 'transparent_pdf_text')
            self.assertEqual(path.read_bytes(), source)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            pages[0].image[0, 0] = 0
            self.assertIsNone(structure.process_text_page(pages[0], Thresholds()))
            path.write_bytes(clean)
            self.assertIsNone(structure.process_text_page(pages[1], Thresholds()))

    def test_file_path_respects_dpi_and_explicit_structure_bypass(self):
        clean, source = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.pdf'; path.write_bytes(source)
            with patch('wm.pipeline.masks.estimate_single_page', side_effect=AssertionError('raster called')):
                got = pipeline.process_file(path, dpi=72, use_ocr=False)
            np.testing.assert_array_equal(got[0].output.image, pixels(clean))
            self.assertEqual(got[0].source.dpi, 72)
            self.assertTrue(got[0].verdict.review_needed)
            with patch('wm.pipeline.structure.process_text_page') as structural, \
                 patch('wm.pipeline.masks.estimate_single_page', return_value=object()), \
                 patch('wm.pipeline._process_one', return_value='raster'):
                got = pipeline.process_file(path, dpi=72, skip_structure=True, use_ocr=False)
            self.assertEqual(got, ['raster'])
            structural.assert_not_called()


if __name__ == '__main__':
    unittest.main()
