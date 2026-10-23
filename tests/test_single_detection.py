import unittest
from unittest.mock import patch
import cv2
import numpy as np
from wm import masks,repeated
from wm.contract import MaskLayer,Thresholds


class PriorTests(unittest.TestCase):
    def test_unrelated_prior_cannot_replace_specific_detection(self):
        a=np.zeros((80,90),np.float32);a[10:15,10:50]=1
        b=np.zeros_like(a);b[40:65,10:70]=1
        own=MaskLayer(a,confidence=1,method='repeated_oblique_text')
        prior=MaskLayer(b,confidence=1,method='crosspage')
        self.assertIs(masks.with_prior(own,prior),own)
        own.method='tonal'
        self.assertIs(masks.with_prior(own,prior),own)

    def test_matching_prior_can_enhance_legacy_single_detection(self):
        a=np.zeros((80,90),np.float32);a[10:15,10:50]=1
        b=a.copy();b[10:15,50:55]=1
        own=MaskLayer(a,confidence=.5)
        prior=MaskLayer(b,confidence=.9,method='crosspage')
        self.assertIs(masks.with_prior(own,prior),prior)


class RepetitionTests(unittest.TestCase):
    def fixture(self,count):
        h,w=750,600
        clean=np.full((h,w,3),255,np.uint8)
        boxes=[]
        text='PRIVATE 2035'
        size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        for y in [220,470][:count]:
            cv2.putText(clean,text,(150,y),cv2.FONT_HERSHEY_SIMPLEX,1,(100,100,100),2,cv2.LINE_AA)
            boxes.append(np.array([[148,y-size[1]-3],[153+size[0],y-size[1]-3],
                                   [153+size[0],y+5],[148,y+5]],np.float32))
        turn=cv2.getRotationMatrix2D((w/2,h/2),-28,1)
        image=cv2.warpAffine(clean,turn,(w,h),borderValue=(255,255,255))
        source_boxes=[cv2.transform(b[None],turn)[0] for b in boxes]
        undo=cv2.getRotationMatrix2D((w/2,h/2),28,1)
        rw=int(abs(undo[0,0])*w+abs(undo[0,1])*h);rh=int(abs(undo[0,0])*h+abs(undo[0,1])*w)
        undo[0,2]+=(rw-w)/2;undo[1,2]+=(rh-h)/2
        upright=[cv2.transform(b[None],undo)[0] for b in source_boxes]
        observations=[[(b,text,.95) for b in source_boxes],[(b,text,.95) for b in upright]]
        return image,observations

    def test_detects_a_different_phrase_from_visual_repetition(self):
        image,observations=self.fixture(2)
        with patch('wm.repeated.ocr.read_lines',side_effect=observations), \
             patch('wm.repeated.ocr.text_zones',return_value=np.zeros(image.shape[:2],bool)):
            mask=repeated.estimate(image)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.method,'repeated_oblique_text')
        self.assertLess(mask.footprint.mean(),.1)
        # Every actual dark character should have substantial coverage nearby.
        ink=image.mean(2)<180
        near=cv2.dilate(mask.footprint.astype(np.uint8),np.ones((3,3),np.uint8))>0
        self.assertGreater((near&ink).sum()/ink.sum(),.8)

    def test_a_single_oblique_label_is_not_evidence_of_repeating_watermark(self):
        image,observations=self.fixture(1)
        with patch('wm.repeated.ocr.read_lines',side_effect=observations), \
             patch('wm.repeated.ocr.text_zones',return_value=np.zeros(image.shape[:2],bool)):
            self.assertIsNone(repeated.estimate(image))

    def test_no_ocr_or_horizontal_only_returns_to_existing_detector(self):
        blank=np.full((100,200,3),255,np.uint8)
        with patch('wm.repeated.ocr.read_lines',return_value=None):
            self.assertIsNone(repeated.estimate(blank))
        box=np.array([[10,10],[150,10],[150,30],[10,30]],np.float32)
        with patch('wm.repeated.ocr.read_lines',return_value=[(box,'NORMAL BODY TEXT',.99)]):
            self.assertIsNone(repeated.estimate(blank))


class TimingTests(unittest.TestCase):
    def test_detection_time_is_included_in_the_verdict_budget(self):
        from wm import pipeline
        from wm.contract import PageImage,FailureCode
        image=np.full((20,30,3),255,np.uint8)
        page=PageImage(image,200)
        mask=MaskLayer(np.ones((20,30),np.float32),confidence=1,
                       veil_strength=np.ones(3,np.float32),veil_offset=np.zeros(3,np.float32))
        with patch('wm.pipeline.repair.repair',return_value=(image.copy(),[])):
            result=pipeline._process_one(page,mask,Thresholds(time_budget_ms=10),mask_ms=20)
        self.assertGreaterEqual(result.verdict.elapsed_ms,20)
        self.assertFalse(result.verdict.conditions['time_budget'])
        self.assertIn(FailureCode.OUT_OF_TIME_BUDGET,result.verdict.failures)

class EdgeFragmentTests(unittest.TestCase):
    def fixture(self):
        th,tw=24,130
        template=np.zeros((th,tw),np.uint8)
        cv2.putText(template,'PRIVATE',(2,19),cv2.FONT_HERSHEY_SIMPLEX,.7,1,2)
        feature=np.zeros((360,360),np.float32)
        valid=np.zeros((360,360),bool);valid[20:300,20:320]=True
        peaks=[(60,50,.99),(60,130,.99),(60,210,.99)]
        for x,y,_ in peaks:
            feature[y:y+th,x:x+tw]=template*150
        truth=np.zeros(valid.shape,bool)
        for x,y in [(315,130),(180,290)]:
            piece=np.zeros(valid.shape,np.float32)
            piece[y:y+th,x:min(x+tw,360)]=template[:,:min(tw,360-x)]*150
            piece*=valid
            truth|=piece>0
            np.maximum(feature,piece,out=feature)
        footer=np.zeros(valid.shape,np.uint8)
        cv2.putText(footer,'12',(25,278),cv2.FONT_HERSHEY_SIMPLEX,.5,150,1)
        feature=np.maximum(feature,footer);feature*=valid
        protected=np.zeros(valid.shape,bool);protected[265:280,22:50]=True
        return feature,valid,template,peaks,truth,protected

    def test_recovers_a_few_strokes_clipped_at_right_and_bottom(self):
        feature,valid,template,peaks,truth,protected=self.fixture()
        result=np.zeros(valid.shape,bool)
        for x,y,_,_,piece in repeated._edge_fragments(feature,valid,template,peaks,protected):
            result[y:y+piece.shape[0],x:x+piece.shape[1]]|=piece>0
        self.assertGreater((result&truth).sum()/truth.sum(),.9)
        self.assertFalse((result&protected).any())
        self.assertFalse((result&~valid).any())

    def test_ocr_protected_boundary_content_is_never_adopted(self):
        feature,valid,template,peaks,truth,protected=self.fixture()
        protected|=cv2.dilate(truth.astype(np.uint8),np.ones((5,5),np.uint8))>0
        for x,y,_,_,piece in repeated._edge_fragments(feature,valid,template,peaks,protected):
            self.assertFalse(((piece>0)&protected[y:y+piece.shape[0],x:x+piece.shape[1]]).any())

    def test_no_repeat_evidence_means_no_boundary_cleanup(self):
        feature,valid,template,peaks,_,protected=self.fixture()
        self.assertEqual(repeated._edge_fragments(feature,valid,template,peaks[:1],protected),[])


if __name__ == '__main__':
    unittest.main()
