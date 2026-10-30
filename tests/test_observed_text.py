import unittest
from unittest.mock import patch
import cv2
import numpy as np
from wm import observed_text, masks
from wm.contract import MaskLayer


class ObservedTextTests(unittest.TestCase):
    def test_repeat_repair_preserves_scene_captions_but_not_oblique_watermark_boxes(self):
        region = np.ones((200, 300), bool)
        caption = np.array([[30, 40], [150, 40], [150, 60], [30, 60]], np.float32)
        watermark = np.array([[60, 150], [180, 80], [190, 98], [70, 168]], np.float32)
        result = observed_text._protect_scene_text(region,
            [(caption, 'STATION', .60), (watermark, 'OVERLAY', .95)])
        self.assertFalse(result[40:61, 30:151].any())
        self.assertTrue(result[130, 100])
        self.assertTrue(result[180, 250])
        np.testing.assert_array_equal(region, np.ones_like(region))

    def test_two_seeds_require_independent_enhanced_repeat_evidence(self):
        image = np.full((600, 800, 3), (100, 140, 185), np.uint8)
        box = np.array([[80, 270], [350, 130], [374, 174], [104, 314]], np.float32)
        lines = [(box, 'SOURCE 42', .98), (box + [30, 160], 'SOURCE 42', .98)]
        with patch('wm.observed_text.ocr.read_lines', return_value=[]) as read:
            self.assertIsNone(observed_text.repeated_photo(image, lines))
            self.assertEqual(read.call_count, 3)
        # Multiple observations at the same place do not confirm a tiled mark.
        with patch('wm.observed_text.ocr.read_lines', return_value=[lines[0]] * 4):
            self.assertIsNone(observed_text.repeated_photo(image, lines))

    def test_dense_verified_strokes_survive_excessive_padding(self):
        strokes = np.zeros((100, 160), np.uint8)
        strokes[10:90:10, 10:150] = 1
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.assertGreater(cv2.dilate(strokes, np.ones((7, 7), np.uint8)).mean(), .4)
        region = observed_text._bounded_repeat_footprint(strokes, matrix, (160, 100))
        self.assertIsNotNone(region)
        self.assertLess(region.mean(), .4)
        self.assertTrue(region[strokes > 0].all())
        self.assertFalse(region[0].any())
        # Padding fallback must never turn an excessive core into permission.
        strokes[:, :100] = 1
        self.assertIsNone(observed_text._bounded_repeat_footprint(strokes, matrix, (160, 100)))

    def test_one_oblique_sign_or_horizontal_captions_cannot_authorize_repeat_erasure(self):
        image=np.full((600,800,3),(100,140,185),np.uint8)
        box=np.array([[80,270],[350,130],[374,174],[104,314]],np.float32)
        horizontal=np.array([[50,70],[450,70],[450,100],[50,100]],np.float32)
        with patch('wm.observed_text.ocr.read_lines',side_effect=AssertionError('insufficient repeat evidence')):
            self.assertIsNone(observed_text.repeated_photo(image,[(box,'ACTUAL SIGN',.98)]))
            self.assertIsNone(observed_text.repeated_photo(image,[(horizontal,'NORMAL CAPTION',.98)]*6))

    def test_no_small_print_or_top_heading_is_not_shadow_watermark(self):
        image=np.full((650,700,3),255,np.uint8)
        cv2.putText(image,'HEADING',(60,100),cv2.FONT_HERSHEY_SIMPLEX,2,(80,120,100),4)
        with patch('wm.observed_text.ocr.read_lines',side_effect=AssertionError('no body evidence')):
            self.assertIsNone(observed_text.shadow_paper(image,[]))
        body=[(np.array([[40,y],[550,y],[550,y+15],[40,y+15]],np.float32),'ordinary printed line',.99) for y in range(180,600,35)]
        top=np.array([[45,50],[550,50],[550,115],[45,115]],np.float32)
        with patch('wm.observed_text.ocr.read_lines',return_value=[(top,'HEADING',.99)]):
            self.assertIsNone(observed_text.shadow_paper(image,body))

    def test_oversized_shadow_detection_and_repair_are_bounded_without_word_templates(self):
        for token in ['DRAFT 82','REVIEW 57']:
            with self.subTest(token=token):
                clean=np.full((650,700,3),255,np.uint8)
                body=[]
                for y in range(110,600,30):
                    cv2.putText(clean,'reading value 2035',(45,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
                    body.append((np.array([[40,y-13],[310,y-13],[310,y+3],[40,y+3]],np.float32),'reading value 2035',.99))
                source=clean.copy()
                cv2.putText(source,token,(65,363),cv2.FONT_HERSHEY_SIMPLEX,2.3,(150,150,150),6,cv2.LINE_AA)
                cv2.putText(source,token,(58,355),cv2.FONT_HERSHEY_SIMPLEX,2.3,(85,125,105),5,cv2.LINE_AA)
                width=cv2.getTextSize(token,cv2.FONT_HERSHEY_SIMPLEX,2.3,5)[0][0]
                large=np.array([[53,299],[65+width,299],[65+width,367],[53,367]],np.float32)
                def read(view):
                    return body if np.array_equal(view,source) else [(large,token,.99)]
                with patch('wm.observed_text.ocr.read_lines',side_effect=read):
                    mask=observed_text.shadow_paper(source,body)
                    self.assertIsNotNone(mask)
                    self.assertEqual(mask.repair_mode, 'shadow_paper_cleanup')
                    self.assertIn('趋于不透明', mask.shadow_evidence['reason'])
                    output=observed_text.restore_shadow(source,mask)
                np.testing.assert_array_equal(output[~mask.footprint],source[~mask.footprint])
                # Broad coloured word pixels on originally blank paper clear;
                # this does not claim overlapping body type is exactly recovered.
                mark=(source.max(2)-source.min(2)>10)&(clean.mean(2)>250)
                self.assertGreater(np.mean(output[mark].mean(1)>240),.90)

    def test_ocr_disabled_and_dot_style_never_enter_new_text_paths(self):
        source=np.full((300,400,3),255,np.uint8)
        with patch('wm.observed_text.repeated_photo',side_effect=AssertionError('not applicable')),patch('wm.observed_text.shadow_paper',side_effect=AssertionError('not applicable')):
            masks.estimate_single_page(source,use_ocr=False)
            masks.estimate_single_page(source,style='dot_grid')

    def test_shadow_repair_never_changes_unselected_paper_or_coloured_mark(self):
        source=np.full((100,160,3),220,np.uint8);source[10:30,100:150]=(180,30,45)
        region=np.zeros((100,160),bool);region[50:75,10:70]=True
        mask=MaskLayer(region.astype(np.float32),footprint=region)
        with patch('wm.observed_text.ocr.read_lines',return_value=[]):output=observed_text.restore_shadow(source,mask)
        np.testing.assert_array_equal(output[~region],source[~region])


if __name__=='__main__':unittest.main()
