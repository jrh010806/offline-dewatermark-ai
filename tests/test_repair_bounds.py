"""The model may rewrite context; only explicitly requested pixels can reach output."""
import unittest
from unittest.mock import patch
import numpy as np
from wm import lama
from wm.contract import Thresholds


class RepairBoundsTests(unittest.TestCase):
    def test_selected_pixels_on_all_four_image_edges_are_actually_repaired(self):
        for shape in [(335, 596), (650, 730), (958, 1276)]:
            with self.subTest(shape=shape):
                image = np.full((*shape, 3), 80, np.uint8)
                region = np.zeros(shape, bool)
                region[[0, -1], :] = True
                region[:, [0, -1]] = True
                with patch('wm.lama._load_session', return_value=object()), \
                     patch('wm.lama._run_tile', side_effect=lambda s, i, m: np.full_like(i, 230)):
                    output = lama.inpaint(image, region)
                self.assertTrue((output[region] > 220).all())
                np.testing.assert_array_equal(output[~region], image[~region])

    def test_context_never_reaches_output_across_native_and_padded_tiles(self):
        rng = np.random.default_rng(19)
        for shape in [(80,130),(335,596),(650,730)]:
            with self.subTest(shape=shape):
                img = rng.integers(0,180,(*shape,3),dtype=np.uint8)
                mask = np.zeros(shape,bool)
                mask[2:12,3:30] = True
                mask[-14:-3,-25:-2] = True
                with patch('wm.lama._load_session',return_value=object()), \
                     patch('wm.lama._run_tile',side_effect=lambda s,i,m: np.full_like(i,230)):
                    out = lama.inpaint(img,mask,Thresholds(lama_context_margin=64))
                np.testing.assert_array_equal(out[~mask],img[~mask])
                self.assertTrue((out[mask]>220).all())

    def test_small_input_is_padded_without_rescaling_its_pixels(self):
        img = np.arange(31*71*3,dtype=np.uint8).reshape(31,71,3)
        mask = np.zeros((31,71),bool);mask[4:9,5:11]=True
        seen=[]
        def model(s,i,m):
            seen.append(i.copy())
            return i.copy()
        with patch('wm.lama._load_session',return_value=object()), \
             patch('wm.lama._run_tile',side_effect=model):
            out = lama.inpaint(img,mask,Thresholds())
        self.assertEqual(seen[0].shape,(512,512,3))
        np.testing.assert_array_equal(seen[0][:31,:71],img)
        np.testing.assert_array_equal(out,img)


class DocumentOverlayTests(unittest.TestCase):
    def test_invertible_document_preserves_existing_ink_without_model_generation(self):
        import cv2
        from wm.contract import MaskLayer,RepairName
        clean=np.full((160,240,3),255,np.uint8)
        cv2.putText(clean,'READ 2035',(30,85),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        alpha=np.zeros(clean.shape[:2],np.float32);alpha[25:125,65:155]=1
        source=np.clip(clean*(1-.8*alpha[...,None])+32*alpha[...,None],0,255).astype(np.uint8)
        mask=MaskLayer(alpha,veil_strength=np.full(3,.2,np.float32),
                       veil_offset=np.full(3,32,np.float32),paper=np.full(clean.shape,255,np.float32),
                       footprint=alpha>0,repair_mode='document_overlay',
                       text_keepout=np.ones(alpha.shape,bool))
        with patch('wm.lama._load_session',side_effect=AssertionError('model must not redraw an invertible page')):
            out,used=lama.repair_with_lama(source,mask,Thresholds())
        np.testing.assert_array_equal(out[alpha==0],source[alpha==0])
        original_ink=clean.mean(2)==0
        self.assertTrue((out[original_ink].mean(1)<5).all())
        self.assertLess(np.abs(out.astype(float)-clean).mean(),.3)  # uint8 inverse quantization
        self.assertEqual(used,[RepairName.ALPHA_UNMIX])

if __name__ == '__main__':
    unittest.main()
