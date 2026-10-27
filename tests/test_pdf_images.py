import unittest
from io import BytesIO

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw

from wm.pdf_images import strip_translucent_images
from wm.pdf_text import strip_translucent_text


def pixels(data):
    with fitz.open(stream=data, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(colorspace=fitz.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)


class PDFImageTests(unittest.TestCase):
    def test_shared_image_is_removed_only_at_overlay_instance(self):
        self.check_shared_image(70)

    def test_stronger_translucent_overlay_is_removed_without_touching_shared_logo(self):
        self.check_shared_image(178)

    def check_shared_image(self, opacity):
        image = Image.new("RGBA", (120, 120))
        ImageDraw.Draw(image).ellipse((5, 5, 115, 115), fill=(40, 80, 150, opacity))
        file = BytesIO(); image.save(file, format="PNG")
        doc = fitz.open(); p = doc.new_page(width=400, height=500)
        for y in range(60, 450, 20):
            p.insert_text((20, y), "Original body content 123456789", fontsize=11)
        xref = p.insert_image((10, 5, 40, 35), stream=file.getvalue())
        clean = doc.tobytes()
        p.insert_image((100, 140, 300, 340), xref=xref)
        source = doc.tobytes(); doc.close()
        output, counts = strip_translucent_images(source)
        self.assertEqual(counts, (1,))
        np.testing.assert_array_equal(pixels(output), pixels(clean))
        with fitz.open(stream=output, filetype="pdf") as result:
            self.assertEqual(len(result[0].get_image_info()), 1)

    def test_opaque_image_and_small_translucent_logo_are_kept(self):
        for alpha, rect in [(255, (100, 140, 300, 340)), (70, (10, 5, 40, 35))]:
            with self.subTest(alpha=alpha):
                image = Image.new("RGBA", (120, 120), (50, 60, 70, alpha))
                file = BytesIO(); image.save(file, format="PNG")
                doc = fitz.open(); p = doc.new_page(width=400, height=500)
                for y in range(60, 450, 20):
                    p.insert_text((20, y), "Original body content 123456789", fontsize=11)
                p.insert_image(rect, stream=file.getvalue())
                source = doc.tobytes(); doc.close()
                output, counts = strip_translucent_images(source)
                self.assertEqual(counts, (0,))
                self.assertEqual(output, source)

    def test_cjk_line_requires_matching_confirmed_overlay(self):
        for colour, expected in [((.2, .3, .6), 2), ((.8, .1, .1), 1)]:
            with self.subTest(colour=colour):
                doc = fitz.open(); p = doc.new_page(width=500, height=600)
                for y in range(40, 560, 20):
                    p.insert_text((20, y), "Original body content 123456789", fontsize=11)
                p.insert_text((70, 300), "Overlay sample", fontsize=40, color=(.2, .3, .6),
                              fill_opacity=.25, morph=(fitz.Point(70, 300), fitz.Matrix(30)))
                p.insert_text((80, 350), "内部演示样本", fontname="china-s", fontsize=32,
                              color=colour, fill_opacity=.25)
                source = doc.tobytes(); doc.close()
                output, counts = strip_translucent_text(source)
                self.assertEqual(counts, (expected,))
                with fitz.open(stream=output, filetype="pdf") as result:
                    self.assertIn("Original body", result[0].get_text())
                    self.assertEqual("内部演示样本" in result[0].get_text(), expected == 1)

    def test_translucent_background_behind_body_is_preserved(self):
        image = Image.new("RGBA", (120, 120), (50, 60, 70, 70))
        file = BytesIO(); image.save(file, format="PNG")
        doc = fitz.open(); p = doc.new_page(width=400, height=500)
        p.insert_image((100, 140, 300, 340), stream=file.getvalue())
        for y in range(60, 450, 20):
            p.insert_text((20, y), "Original body content 123456789", fontsize=11)
        source = doc.tobytes(); doc.close()
        output, counts = strip_translucent_images(source)
        self.assertEqual(counts, (0,))
        self.assertEqual(output, source)

    def test_full_page_sparse_overlay_preserves_raster_underlay(self):
        import cv2
        base = np.full((500, 400, 3), 245, np.uint8)
        for y in range(50, 460, 25):
            cv2.putText(base, "Body content 123", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (30, 30, 30), 1)
        overlay = np.zeros((500, 400, 4), np.uint8)
        support = np.zeros((500, 400), np.uint8)
        cv2.putText(support, "DRAFT 401", (30, 260), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, 255, 3)
        overlay[..., :3] = (40, 80, 150)
        overlay[..., 3] = np.uint8(support.astype(float)*90/255)
        raw = BytesIO(); Image.fromarray(base).save(raw, format="PNG")
        layer = BytesIO(); Image.fromarray(overlay).save(layer, format="PNG")
        doc = fitz.open(); p = doc.new_page(width=400, height=500)
        p.insert_image(p.rect, stream=raw.getvalue())
        clean = doc.tobytes()
        p.insert_image(p.rect, stream=layer.getvalue())
        source = doc.tobytes(); doc.close()
        output, counts = strip_translucent_images(source)
        self.assertEqual(counts, (1,))
        np.testing.assert_array_equal(pixels(output), pixels(clean))
