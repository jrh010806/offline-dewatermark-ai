import unittest
from unittest.mock import patch
import cv2
import numpy as np

from wm import paper_repeat, repair
from wm.contract import Thresholds


class PaperRepeatTests(unittest.TestCase):
    def fixture(self, opacity, text, positions=None):
        h,w=900,1100
        clean=np.full((h,w,3),250,np.uint8)
        keepout=np.zeros((h,w),np.uint8)
        for y in [180,215,250,410,445,480]:
            cv2.putText(clean,'Body 4072 normal content',(130,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),2,cv2.LINE_AA)
            cv2.rectangle(keepout,(125,y-22),(530,y+5),1,-1)
        cv2.putText(clean,'GRAY TEXT',(130,620),cv2.FONT_HERSHEY_SIMPLEX,.8,(150,150,150),2)
        cv2.rectangle(keepout,(120,590),(310,630),1,-1)
        cv2.circle(clean,(820,740),50,(180,20,30),4)
        marks=np.zeros((h,w),np.uint8);boxes=[]
        scale=.68;thickness=2
        tw,th=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)[0]
        for x,y in positions or [(x,y) for y in [100,250,400,550,700,850] for x in [80,350,620,890]]:
            cv2.putText(marks,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,255,thickness,cv2.LINE_AA)
            boxes.append(np.array([[x-2,y-th-4],[x+tw+4,y-th-4],[x+tw+4,y+5],[x-2,y+5]],np.float32))
        turn=cv2.getRotationMatrix2D((w/2,h/2),-27,1)
        alpha=cv2.warpAffine(marks,turn,(w,h)).astype(np.float32)/255*opacity
        source=np.uint8(clean*(1-alpha[:,:,None])+40*alpha[:,:,None])
        source_boxes=[cv2.transform(b[None],turn)[0] for b in boxes]
        undo=cv2.getRotationMatrix2D((w/2,h/2),27,1)
        rw=int(abs(undo[0,0])*w+abs(undo[0,1])*h);rh=int(abs(undo[0,0])*h+abs(undo[0,1])*w)
        undo[0,2]+=(rw-w)/2;undo[1,2]+=(rh-h)/2
        upright=[cv2.transform(b[None],undo)[0] for b in source_boxes]
        observations=[[(b,text,.99) for b in source_boxes],[(b,text,.99) for b in upright],[(b,text,.99) for b in upright]]
        return clean,source,alpha,keepout.astype(bool),observations

    def test_more_than_twelve_copies_without_generative_repair_and_preserve_ink(self):
        for opacity,text in [(.25,'PRIVATE 4072'),(.72,'REVIEW 5183')]:
            with self.subTest(opacity=opacity):
                clean,source,alpha,zones,observations=self.fixture(opacity,text)
                with patch('wm.paper_repeat.ocr.read_lines',side_effect=observations), \
                     patch('wm.paper_repeat.ocr.text_zones',return_value=zones):
                    mask=paper_repeat.estimate(source)
                self.assertIsNotNone(mask)
                region=alpha>.03
                self.assertGreater(np.mean(mask.footprint[region]),.90)
                with patch('wm.lama._load_session',side_effect=AssertionError('must not generate paper glyphs')):
                    output,_=repair.repair(source,mask,Thresholds())
                body=clean.mean(2)<170
                # 反混合路径允许改动被水印覆盖的正文，验收用量化的损伤预算：
                # 绝大多数正文必须逐字节一致，超过 12 灰度级的损伤不可超过 2.5%，
                # 且整页必须明显更接近干净原稿；禁止生成式修复的约束不变。
                same=(output[body]==source[body]).mean()
                self.assertGreaterEqual(same,.90)
                worse=(abs(output.astype(float)-clean)>abs(source.astype(float)-clean)+12)[body].mean()
                self.assertLess(worse,.025)
                mae_before=abs(source.astype(float)-clean).mean()
                mae_after=abs(output.astype(float)-clean).mean()
                self.assertLess(mae_after,mae_before*.2)
                np.testing.assert_array_equal(output[~mask.footprint],source[~mask.footprint])
                paper=region & ~cv2.dilate(body.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                before=abs(source.astype(float)-clean)[paper].mean()
                after=abs(output.astype(float)-clean)[paper].mean()
                self.assertLess(after,before*.15)

    def test_sparse_ocr_parts_are_combined_instead_of_only_erasing_a_prefix(self):
        clean,source,alpha,zones,observations=self.fixture(.25,'REVIEW 4072',[(230,280),(620,280),(230,680),(620,680)])
        parts=[]
        for b,_,conf in observations[1]:
            x0,y0=b.min(0);x1,y1=b.max(0);mid=(x0+x1)/2
            for lo,hi,text in [(x0,mid,'REVIEW'),(mid,x1,'4072')]:
                parts.append((np.array([[lo,y0],[hi,y0],[hi,y1],[lo,y1]],np.float32),text,conf))
        with patch('wm.paper_repeat.ocr.read_lines',side_effect=[observations[0],parts,parts]), \
             patch('wm.paper_repeat.ocr.text_zones',return_value=zones):
            mask=paper_repeat.estimate(source)
        self.assertIsNotNone(mask)
        self.assertGreater(mask.footprint[alpha>.03].mean(),.9)

    def test_normal_horizontal_body_and_photographs_are_declined(self):
        clean,_,_,_,_=self.fixture(.25,'PRIVATE 4072')
        horizontal=np.array([[120,140],[530,140],[530,180],[120,180]],np.float32)
        with patch('wm.paper_repeat.ocr.read_lines',return_value=[(horizontal,'Normal body',.99)]):
            self.assertIsNone(paper_repeat.estimate(clean))
        photo=np.full((300,400,3),120,np.uint8)
        with patch('wm.paper_repeat.ocr.read_lines',side_effect=AssertionError('no paper')):
            self.assertIsNone(paper_repeat.estimate(photo))


class TableRuleTests(unittest.TestCase):
    def test_uncertain_overlay_fit_does_not_whiten_a_continuous_table_rule(self):
        from wm.contract import MaskLayer
        clean=np.full((220,500,3),245,np.uint8)
        clean[110,30:470]=0
        source=clean.copy()
        source[95:125,150:330]=np.uint8(source[95:125,150:330].astype(float)*.2+144)
        region=np.zeros(source.shape[:2],bool);region[95:125,150:330]=True
        mask=MaskLayer(region.astype(np.float32)*.3,confidence=1.,
            veil_strength=np.full(3,.2,np.float32),veil_offset=np.full(3,144,np.float32),
            paper=np.full(source.shape,245,np.float32),footprint=region,repair_mode='document_overlay')
        output,_=repair.repair(source,mask,Thresholds())
        self.assertTrue(np.all(output[110,30:470].mean(1)<190))
        np.testing.assert_array_equal(output[~region],source[~region])

if __name__=='__main__':unittest.main()
