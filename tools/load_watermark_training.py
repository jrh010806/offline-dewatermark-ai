"""Minimal NumPy/Pillow loader for the paired training corpus (no torch needed).

Paths are relative to dataset root. Alpha/mask are returned separately; an
alpha channel is never substituted for a binary support mask. No implicit
resize or JPEG round trip changes the reference pixels.
"""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image

def iter_pairs(root,split='train',kind='image',include_negatives=True):
    root=Path(root)
    with (root/f'splits/{split}.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['kind']!=kind or (row['is_negative'] and not include_negatives):continue
            if kind=='image':
                yield dict(row=row,**read_pixels(root,row['paths']))
            else:
                for page in row['pages']:
                    yield dict(row=row,page=page['page'],**read_pixels(root,page['paths']))

def read_pixels(root,paths):
    out={}
    for name in ['watermarked','clean','alpha','mask']:
        mode='RGB' if name in ['watermarked','clean'] else 'L'
        out[name]=np.asarray(Image.open(root/paths[name]).convert(mode)).astype('float32')/255.
    if out['watermarked'].shape != out['clean'].shape:raise ValueError('Pair dimension mismatch')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'data/watermark_training_v1')
    p.add_argument('--split',choices=['train','validation','test'],default='train');p.add_argument('--kind',choices=['image','pdf'],default='image')
    p.add_argument('--limit',type=int,default=3);a=p.parse_args()
    for i,example in enumerate(iter_pairs(a.root,a.split,a.kind)):
        print(example['row']['id'],example.get('page',''),example['watermarked'].shape,'mask pixels',int(example['mask'].sum()))
        if i+1>=a.limit:break

if __name__=='__main__':main()
