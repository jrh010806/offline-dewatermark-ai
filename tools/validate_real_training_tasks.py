"""Independently validate task labels, source isolation and loader usability."""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from load_manual_increment import load_training_sample, ManualTrainingDataset

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/watermark_training_increment_v1_1'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def validate():
    records=read(D/'training_tasks.jsonl')
    strict=read(D/'manifest.jsonl')
    original=read(ROOT/'data/watermark_training_v1/manifest.jsonl')
    groups={}
    for r in original:
        if r['source_group'] in groups:assert groups[r['source_group']]==r['split']
        groups[r['source_group']]=r['split']
    visual=json.loads((D/'qa/real_training_visual_review.json').read_text())
    assert visual['status']=='passed' and len(visual['samples'])==8
    assert len(records)==129 and len({r['id'] for r in records})==129
    assert len(strict)==121
    checked_files=set();new=[];coco={'images':[],'annotations':[],
                                   'categories':[{'id':1,'name':'watermark'}]}
    for r in records:
        assert r['training_admission'] is True and r['split']=='train'
        assert r['source_group']==r['split_group']
        assert groups.get(r['source_group'],'train')=='train',r['id']
        for role,p in r['paths'].items():
            path=(D/p).resolve();assert path.is_relative_to(D.resolve()) and path.is_file(),p
            checked_files.add(p)
            if 'sha256' in r:assert digest(path)==r['sha256'][role],(r['id'],role)
        w,h=r['size']
        assert Image.open(D/r['paths']['input']).size==(w,h)
        if r['category']!='strict_pair':
            new.append(r)
            v=next(v for v in visual['samples'] if v['id']==r['id'])
            assert digest(D/r['paths']['input'])==v['input_sha256']
            for role,p in r['paths'].items():assert digest(D/p)==v['path_sha256'][role]
            assert 'alpha' not in r['paths'] and r['alpha_available'] is False
        for task in r['tasks']:
            sample=load_training_sample(D,r,task)
            assert sample['input'].shape==(3,h,w)
            assert np.isfinite(sample['input']).all() and 0<=sample['input'].min()<=sample['input'].max()<=1
            if task=='restoration':
                assert sample['target'].shape==(3,h,w)
                assert sample['loss_weight'].shape==(1,h,w)
                assert np.isfinite(sample['target']).all() and np.isfinite(sample['loss_weight']).all()
                assert (sample['loss_weight']>0).sum()>w*h*.7
                if r['category']!='strict_pair':
                    assert r['sample_weight']==.25
                    assert sample['loss_weight'].max()<=.25
                    assert not r['clean_is_same_capture_ground_truth']
            else:
                boxes=sample['boxes'];assert boxes.shape[1]==4
                if len(boxes):
                    assert (boxes[:,0]>=0).all() and (boxes[:,1]>=0).all()
                    assert (boxes[:,2]<=w).all() and (boxes[:,3]<=h).all()
                    assert (boxes[:,2]>boxes[:,0]).all() and (boxes[:,3]>boxes[:,1]).all()
                if 'segmentation' in sample:
                    seg=sample['segmentation'];weights=sample['segmentation_weight']
                    assert seg.shape==weights.shape==(1,h,w)
                    assert set(np.unique(seg))<={0,1}
                    assert (weights>=0).all() and (weights<=1).all()
                    if r['category']=='red_pen_segmentation':
                        labels=np.array(Image.open(D/r['paths']['segmentation']))
                        assert set(np.unique(labels))=={0,1,255}
                        assert not weights[0][labels==255].any()
                elif r['category']=='paper_scan_box_detection':
                    assert r['segmentation_available'] is False and r['restoration_target_available'] is False
                    assert r['qa']['inliers']>=60 and r['qa']['median_error_px']<1.6
                image_id=len(coco['images'])+1
                coco['images'].append(dict(id=image_id,file_name=r['paths']['input'],width=w,height=h,
                                           source_group=r['source_group'],split='train',sample_id=r['id'],
                                           label_quality=r['label_quality'],sample_weight=r['sample_weight']))
                for x1,y1,x2,y2 in boxes.tolist():
                    coco['annotations'].append(dict(id=len(coco['annotations'])+1,image_id=image_id,
                                                   category_id=1,bbox=[x1,y1,x2-x1,y2-y1],
                                                   area=(x2-x1)*(y2-y1),iscrowd=0))
    assert len(new)==8
    assert collections.Counter(r['category'] for r in new)=={
        'approximate_restoration_pair':3,'paper_scan_box_detection':4,'red_pen_segmentation':1}
    assert not (ROOT/'data/manual_collections/douyin_batch_001').exists()
    coco_path=D/'detection_coco_train.json';coco_path.write_text(json.dumps(coco,ensure_ascii=False))
    summary=dict(status='passed',unique_training_records=129,strict_pairs=121,
                 new_real_training_records=8,approximate_restoration_pairs=3,
                 detection_only_images=5,restoration_records=124,detection_records=126,
                 real_source_groups=3,all_records_split='train',source_split_leakage=False,
                 checked_file_references=len(checked_files),loader_task_reads=250,
                 fake_clean_targets_created=0,fake_alpha_created=0,
                 projected_scan_masks_used_as_precise_segmentation=False,
                 original_strict_manifest_sha256=digest(D/'manifest.jsonl'),
                 task_manifest_sha256=digest(D/'training_tasks.jsonl'),
                 visual_review='qa/real_training_visual_review.json',
                 quality_limit='3 restoration pairs are approximate and weighted .25; 4 scan labels are boxes only; red image has no restoration target')
    path=D/'qa/training_tasks_validation.json';path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # Exercise the actual Dataset API including the last real input in each
    # task, plus a tiny forward/backward pass with a throwaway module.
    restore=ManualTrainingDataset(D,'restoration');detect=ManualTrainingDataset(D,'detection')
    assert len(restore)==124 and len(detect)==126
    assert restore[-1]['label_quality'].startswith('registered_photographic')
    assert detect[-1]['boxes'].shape==(1,4)
    try:
        import torch
        from torch.utils.data import DataLoader,Subset
        batch=next(iter(DataLoader(Subset(restore,[len(restore)-1]),batch_size=1)))
        model=torch.nn.Conv2d(3,3,1)
        inp=batch['input'][:,:,:128,:128];target=batch['target'][:,:,:128,:128]
        weight=batch['loss_weight'][:,:,:128,:128]
        pred=model(inp)
        loss=((pred-target).abs()*weight).sum()/(3*(weight>0).sum()).clamp_min(1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.weight.grad).all()
        assert model.weight.grad.abs().sum()>0
        summary['torch_data_loader_and_throwaway_gradient_check']='passed; no application model trained or weights saved'
    except ImportError:
        summary['torch_data_loader_and_throwaway_gradient_check']='torch not installed; numpy loader passed'
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=False,indent=2))
