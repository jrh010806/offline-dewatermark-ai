import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import contrast_text, masks, paper_repeat


class ContrastTextTests(unittest.TestCase):
    def oblique(self, count=1, color=(30,30,30), background=245):
        image=np.full((600,700,3),background,np.uint8)
        boxes=[]
        for y in [240,430][:count]:
            text='DRAFT 46';size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.8,3)[0]
            cv2.putText(image,text,(130,y),cv2.FONT_HERSHEY_SIMPLEX,1.8,color,3,cv2.LINE_AA)
            boxes.append(np.array([[126,y-size[1]-5],[138+size[0],y-size[1]-5],
                                   [138+size[0],y+7],[126,y+7]],np.float32))
        matrix=cv2.getRotationMatrix2D((350,300),25,1)
        image=cv2.warpAffine(image,matrix,(700,600),borderValue=(background,)*3)
        lines=[(cv2.transform(b[None],matrix)[0],'DRAFT 46',.95) for b in boxes]
        return image,lines

    def test_dark_oblique_strokes_are_selected_without_whole_box_erasure(self):
        image,lines=self.oblique()
        mask=contrast_text.dark_oblique(image,lines)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.repair_mode,'inpaint')
        ink=image.mean(2)<100
        self.assertGreater((mask.footprint&ink).sum()/ink.sum(),.97)
        self.assertLess(mask.footprint.sum(),cv2.contourArea(lines[0][0])*.65)

    def test_horizontal_body_type_and_colored_chart_elements_are_not_dark_overlays(self):
        image,lines=self.oblique(color=(220,30,30))
        self.assertIsNone(contrast_text.dark_oblique(image,lines))
        box=np.array([[50,80],[600,80],[600,130],[50,130]],np.float32)
        self.assertIsNone(contrast_text.dark_oblique(image,[(box,'ORDINARY HEADING',.99)]))

    def test_bright_cleanup_requires_distinct_repeated_oblique_geometry(self):
        image,lines=self.oblique(2,color=(240,240,240),background=175)
        with patch('wm.contrast_text.ocr.read_lines',return_value=lines[:1]):
            self.assertIsNone(contrast_text.bright_repeated(image))
        with patch('wm.contrast_text.ocr.read_lines',return_value=[lines[0],lines[0]]):
            self.assertIsNone(contrast_text.bright_repeated(image))
        with patch('wm.contrast_text.ocr.read_lines',return_value=lines):
            mask=contrast_text.bright_repeated(image)
        self.assertIsNotNone(mask)
        with patch('wm.lama.inpaint',side_effect=lambda src,region,thr:
                   np.where(region[...,None],175,src).astype(np.uint8)):
            restored=contrast_text.restore_bright(image,mask)
        np.testing.assert_array_equal(restored[~mask.footprint],image[~mask.footprint])
        bright=image.mean(2)>220
        self.assertLess(restored[bright].mean(),image[bright].mean()-30)

    def test_brightened_print_is_preserved_instead_of_filled_as_paper(self):
        from wm.contract import MaskLayer
        clean=np.full((100,160,3),175,np.uint8)
        cv2.putText(clean,'READ',(15,57),cv2.FONT_HERSHEY_SIMPLEX,1,(35,35,35),2,cv2.LINE_AA)
        source=np.rint(clean*.55+255*.45).astype(np.uint8)
        domain=np.zeros((100,160),bool);domain[10:85,5:150]=True
        mask=MaskLayer(domain.astype(np.float32),footprint=domain,paper=np.full(clean.shape,175,np.float32))
        with patch('wm.lama.inpaint',side_effect=lambda src,region,thr:
                   np.where(region[...,None],175,src).astype(np.uint8)):
            restored=contrast_text.restore_bright(source,mask)
        ink=clean.mean(2)<70
        np.testing.assert_array_equal(restored[ink],source[ink])
        np.testing.assert_array_equal(restored[~domain],source[~domain])

    def test_white_stroke_edges_do_not_turn_into_dark_negative_outlines(self):
        from wm.contract import MaskLayer
        image=np.full((100,180,3),175,np.uint8)
        cv2.putText(image,'WIDE',(10,65),cv2.FONT_HERSHEY_SIMPLEX,1.7,(225,225,225),4,cv2.LINE_AA)
        domain=np.zeros((100,180),bool);domain[8:90,3:175]=True
        mask=MaskLayer(domain.astype(np.float32),footprint=domain,paper=np.full(image.shape,175,np.float32))
        with patch('wm.lama.inpaint',side_effect=lambda src,region,thr:
                   np.where(region[...,None],175,src).astype(np.uint8)):
            result=contrast_text.restore_bright(image,mask)
        self.assertGreaterEqual(int(result[domain].min()),175)
        self.assertEqual(int(result[domain].max()),175)

    def test_specific_contrast_detection_cannot_be_replaced_by_crosspage_prior(self):
        from wm.contract import MaskLayer
        region=np.zeros((80,100),np.float32);region[10:40,20:70]=1
        own=MaskLayer(region,confidence=.9,method='contrast_bright_repeated')
        prior=MaskLayer(region.copy(),confidence=1,method='crosspage')
        self.assertIs(masks.with_prior(own,prior),own)

    def test_original_readable_photo_caption_is_excluded(self):
        image=np.full((400,600,3),(90,100,20),np.uint8)
        cv2.putText(image,'KEEP 2035',(180,210),cv2.FONT_HERSHEY_SIMPLEX,.9,(105,115,20),2)
        box=np.array([[178,180],[360,180],[360,214],[178,214]],np.float32)
        source=[(box,'KEEP 2035',.99)]
        with patch('wm.contrast_text.ocr.read_lines',return_value=source):
            self.assertIsNone(contrast_text.faint_chromatic(image,source))

    def test_disabling_ocr_does_not_call_contrast_detector(self):
        image=np.full((300,400,3),255,np.uint8)
        with patch('wm.contrast_text.estimate',side_effect=AssertionError('disabled')):
            masks.estimate_single_page(image,use_ocr=False)

    def test_locked_dot_style_cannot_select_text_cleanup(self):
        image=np.full((300,400,3),255,np.uint8)
        with patch('wm.contrast_text.estimate',side_effect=AssertionError('wrong style')):
            masks.estimate_single_page(image,style='dot_grid')


if __name__=='__main__':unittest.main()
