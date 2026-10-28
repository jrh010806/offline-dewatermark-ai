"""Evaluate the official WDNet pretrained generator on the same eight inputs.
Download URLs and hashes are stored alongside outputs; no labels enter inference.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
import probe_removal_strategies as base
ROOT=base.ROOT
MODEL_DIR=ROOT/'out/strategy_probe/models/wdnet'
sys.path.insert(0,str(MODEL_DIR))

def prepare_model():
 import urllib.request
 files={
  'WDNet_G.pkl':('https://drive.usercontent.google.com/download?id=1Tv0iM3ZbM0j3akP9uI1myC18IUnyDNjL&export=download&confirm=t','0156ef376692c911e4804d7226656c9fd395c84ff63ef99411e7b4d5b779686f'),
  'WDNet.py':('https://raw.githubusercontent.com/MRUIL/WDNet/main/WDNet.py','d5f8fac2bf45b09c34a56f8c98900fac3f79622868ead4af272e1eb6a34bcd0f'),
  'unet_parts.py':('https://raw.githubusercontent.com/MRUIL/WDNet/main/unet_parts.py','6aab7316bf0a0f413d2b58a76bd15484af5a526d2dd200d1a21cc169e9691010')}
 MODEL_DIR.mkdir(parents=True,exist_ok=True)
 for name,(url,digest) in files.items():
  path=MODEL_DIR/name
  if not path.exists():
   with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
   assert __import__('hashlib').sha256(data).hexdigest()==digest, 'Downloaded asset changed: '+name
   path.write_bytes(data)
  assert base.sha(path)==digest,'Asset hash mismatch: '+name
 s=(MODEL_DIR/'WDNet.py').read_text();gen=s[s.index('class generator'):s.index('class discriminator')]
 gen=gen.replace('return I,mask,a,w,I_watermark','return I,mask,a,w,I_watermark,I_watermark2')
 (MODEL_DIR/'inference_generator.py').write_text('# Official generator; training imports omitted; refined background exposed.\nimport torch\nimport torch.nn as nn\nfrom unet_parts import *\n'+gen)
 print('WDNet assets verified',flush=True)

def predict(src,model):
 h,w=src.shape[:2]
 def infer(img):
  x=torch.from_numpy(img.transpose(2,0,1).copy()).float()[None].to('mps')/255
  with torch.inference_mode(): _,p,a,color,raw,b=model(x)
  return b[0].cpu().numpy().transpose(1,2,0)*255,p[0,0].cpu().numpy(),raw[0].cpu().numpy().transpose(1,2,0)*255
 t=time.perf_counter();small=cv2.resize(src,(256,256),interpolation=cv2.INTER_AREA);b,p,raw=infer(small)
 prob=cv2.resize(p,(w,h));bg=cv2.resize(b,(w,h));glob=base.quant(bg*prob[...,None]+src*(1-prob[...,None]));global_seconds=time.perf_counter()-t
 t=time.perf_counter();acc=np.zeros_like(src,dtype=np.float32);raw_acc=acc.copy();weight=np.zeros((h,w),np.float32);pacc=weight.copy()
 yy=sorted(set(list(range(0,max(1,h-255),192))+[max(0,h-256)]));xx=sorted(set(list(range(0,max(1,w-255),192))+[max(0,w-256)]))
 win=np.maximum(np.outer(np.hanning(256),np.hanning(256)),.02).astype(np.float32)
 for y in yy:
  for x in xx:
   crop=src[y:y+256,x:x+256];ch,cw=crop.shape[:2];inp=cv2.copyMakeBorder(crop,0,256-ch,0,256-cw,cv2.BORDER_REFLECT_101)
   b,p,raw=infer(inp);z=win[:ch,:cw];p=p[:ch,:cw];b=b[:ch,:cw];raw=raw[:ch,:cw]
   acc[y:y+ch,x:x+cw]+=(b*p[...,None]+crop*(1-p[...,None]))*z[...,None]
   raw_acc[y:y+ch,x:x+cw]+=raw*z[...,None];weight[y:y+ch,x:x+cw]+=z;pacc[y:y+ch,x:x+cw]+=p*z
 secs=time.perf_counter()-t
 return {'wdnet_global':glob,'wdnet_tiles':base.quant(acc/weight[...,None]),'wdnet_unmix_tiles':base.quant(raw_acc/weight[...,None])}, {'wdnet_global':prob>.5,'wdnet_tiles':pacc/weight>.5,'wdnet_unmix_tiles':pacc/weight>.5}, {'wdnet_global':global_seconds,'wdnet_tiles':secs,'wdnet_unmix_tiles':secs}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path,nargs='?');ap.add_argument('--prepare-model',action='store_true');args=ap.parse_args()
 if args.prepare_model:prepare_model();return
 if args.folder is None:ap.error('folder is required for inference')
 from inference_generator import generator
 out=args.folder.resolve();path=out/'wdnet_results.json'
 if path.exists():raise SystemExit('Refusing to overwrite existing WDNet result')
 torch.set_num_threads(4);cv2.setNumThreads(4)
 t=time.perf_counter();model=generator(3,3);model.load_state_dict(torch.load(MODEL_DIR/'WDNet_G.pkl',map_location='cpu',weights_only=True),strict=True);model=model.eval().to('mps');load_s=time.perf_counter()-t
 provenance=dict(source='https://github.com/MRUIL/WDNet',weights_source='https://drive.google.com/file/d/1Tv0iM3ZbM0j3akP9uI1myC18IUnyDNjL/view',device='mps',model_load_seconds=load_s,weights_sha256=base.sha(MODEL_DIR/'WDNet_G.pkl'),script_sha256=base.sha(Path(__file__)),source_sha256={p.name:base.sha(p) for p in MODEL_DIR.glob('*.py')},notes='Official linked contributor checkpoint (50 epochs), strict state-dict load, no finetuning. Generator unchanged except exposing final refined background for blending. Global resize 256; native tiles256 stride192 feather; raw uses predicted alpha/color algebraic decomposition before refinement. No truth in inference. Native refine/raw share inference time. No original-claim reproduction.')
 (out/'wdnet_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2));rows=[]
 cases={r['case_id']:r for r in map(json.loads,(base.DATA/'manifest.jsonl').read_text().splitlines())}
 for cid in base.IDS:
  case=cases[cid];src=base.load(base.DATA/case['paths']['input']);preds,masks,timings=predict(src,model)
  # Reference loaded only once all three predictions have finished.
  clean=alpha=None
  if case['score_mode']=='paired':
   clean=base.load(base.DATA/case['paths']['clean']);alpha=np.array(Image.open(base.DATA/case['paths']['alpha']).convert('L'),float)/255
  row=dict(case_id=cid,methods={})
  for k,v in preds.items():
   base.save(out/cid/(k+'.png'),v);base.save(out/cid/(k+'_mask.png'),masks[k]*255)
   row['methods'][k]=base.scores(src,v,clean,alpha,masks[k],timings[k]) if clean is not None else dict(seconds=timings[k],probe_pass=None,manual_review=True)
  rows.append(row);path.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False));print(cid,{k:round(v['seconds'],2) for k,v in row['methods'].items()},flush=True)
if __name__=='__main__':main()
