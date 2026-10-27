"""Portable task-aware reader; legacy records/load_pair retain strict semantics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

def records(root, kind=None):
    root=Path(root)
    for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r.get('training_admission') is True and (kind is None or r['kind']==kind):
            yield r

def load_pair(root, row):
    root=Path(root);p=row['paths']
    return {role:np.array(Image.open(root/p[role]).convert('RGB' if role in ['clean','watermarked'] else 'L'))
            for role in ['watermarked','clean','mask','alpha']}

def training_records(root, task=None):
    """Each image/page occurs once. Select only records supporting the task."""
    root=Path(root)
    qa=json.loads((root/'qa/training_tasks_validation.json').read_text())
    if qa.get('status')!='passed':
        raise ValueError('Task-aware training validation has not passed')
    if hashlib.sha256((root/'training_tasks.jsonl').read_bytes()).hexdigest()!=qa['task_manifest_sha256']:
        raise ValueError('Task manifest changed after validation; rerun validation before training')
    for line in (root/'training_tasks.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row.get('training_admission') is True and (task is None or task in row['tasks']):
            yield row

def load_training_sample(root, row, task):
    """Return float CHW input and task-specific labels, never invented alpha.

    Restoration: target CHW and loss_weight 1HW. The weight includes the
    record's sample_weight; approximate pairs have weight .25.
    Detection: boxes Nx4, optional segmentation 1HW and segmentation_weight
    1HW. No segmentation key means box supervision only. Unknown label edges
    use weight zero. Crop/resize input and labels together during augmentation.
    """
    if task not in row['tasks']:
        raise ValueError(f"{row['id']} does not support {task}")
    root=Path(root);p=row['paths']
    inp=np.asarray(Image.open(root/p['input']).convert('RGB'),dtype=np.float32)/255
    h,w=inp.shape[:2]
    out={'id':row['id'],'input':np.ascontiguousarray(inp.transpose(2,0,1)),
         'source_group':row['source_group'],'label_quality':row['label_quality']}
    sample_weight=float(row['sample_weight'])
    if task=='restoration':
        target=np.asarray(Image.open(root/p['target']).convert('RGB'),dtype=np.float32)/255
        weight=(np.asarray(Image.open(root/p['loss_weight']).convert('L'),dtype=np.float32)/255
                if 'loss_weight' in p else np.ones((h,w),np.float32))
        out.update(target=np.ascontiguousarray(target.transpose(2,0,1)),
                   loss_weight=weight[None]*sample_weight)
    elif task=='detection':
        if 'mask' in p:
            mask=np.asarray(Image.open(root/p['mask']).convert('L'))>0
            valid=np.ones((h,w),np.float32)
            if mask.any():
                yy,xx=np.where(mask);boxes=[[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]]
            else:boxes=[]
            out.update(segmentation=mask.astype(np.float32)[None],
                       segmentation_weight=valid[None]*sample_weight)
        else:
            boxes=row.get('boxes_xyxy',[])
            if row.get('segmentation_available'):
                labels=np.asarray(Image.open(root/p['segmentation']).convert('L'))
                out.update(segmentation=(labels==1).astype(np.float32)[None],
                           segmentation_weight=(labels!=255).astype(np.float32)[None]*sample_weight)
        out.update(boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                   labels=np.ones(len(boxes),dtype=np.int64),box_sample_weight=sample_weight)
    return out

class ManualTrainingDataset:
    """Compatible with torch.utils.data.DataLoader; torch is optional here."""
    def __init__(self, root, task='restoration'):
        self.root=Path(root);self.task=task
        self.rows=list(training_records(self.root,task))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return load_training_sample(self.root,self.rows[index],self.task)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--kind',choices=['image','pdf_page']);ap.add_argument('--limit',type=int,default=3)
    ap.add_argument('--view',choices=['strict','training'],default='training',
                    help='training selects the unified task-aware manifest; strict keeps the original 121 pairs')
    ap.add_argument('--task',choices=['restoration','detection'])
    args=ap.parse_args();total=0
    if args.kind:args.view='strict'
    if args.view=='training':
        if args.kind:ap.error('--kind applies only to --view strict')
        for row in training_records(args.root,args.task):
            task=args.task or row['tasks'][0]
            if total<args.limit:
                data=load_training_sample(args.root,row,task)
                print(json.dumps({'id':row['id'],'task':task,'quality':row['label_quality'],
                                  'shapes':{k:list(v.shape) for k,v in data.items() if isinstance(v,np.ndarray)}}))
            total+=1
        print(json.dumps({'eligible_records':total,'task':args.task or 'all_tasks_unique_records','split':'train'}))
        return
    if args.task:ap.error('--task requires --view training')
    for r in records(args.root,args.kind):
        if total<args.limit:
            data=load_pair(args.root,r)
            print(json.dumps({'id':r['id'],'source_group':r['source_group'],'kind':r['kind'],
                              'shapes':{k:list(v.shape) for k,v in data.items()}},ensure_ascii=False))
        total+=1
    print(json.dumps({'eligible_records':total,'weak_pairs_included':False,'split':'train'}))

if __name__=='__main__':main()
