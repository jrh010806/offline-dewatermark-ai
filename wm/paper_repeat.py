"""Remove repeated neutral oblique overlays on bright paper from one image.

Whole-word visual matches confirm OCR candidates; all matching copies are used.
Only confirmed strokes on paper are filled. Dark print, coloured ink and table
rules are preserved rather than handed to a generative repair model.
"""
from __future__ import annotations
import warnings
import cv2
import numpy as np
from . import ocr, repeated
from .contract import MaskLayer, MaskSource


def table_rules(img):
    gray = img.astype(np.float32).mean(2)
    length = max(25, min(100, min(gray.shape)//12))
    ink = (gray < .65*float(np.percentile(gray, 97))).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    # A dark watermark block is not a table rule. Keep only thin runs;
    # otherwise an invertible dark overlay would be copied back into the result.
    horizontal &= 1-cv2.morphologyEx(horizontal,cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
    vertical &= 1-cv2.morphologyEx(vertical,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
    return cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0


def estimate(img):
    gray = img.astype(np.float32).mean(2)
    if np.mean(gray > 235) < .70:
        return None
    observations = ocr.read_lines(img)
    angles = []
    upright_body = False
    for box, text, confidence in observations or []:
        height, length, tilt = ocr._geometry(box)
        upright_body |= confidence>=.65 and len(text)>=4 and tilt<10
        v = box[1]-box[0]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        if confidence >= .65 and len(text) >= 5 and length > 3*height and 15 < abs(angle) < 60:
            angles.append(angle)
    if not angles:
        return None
    angle = float(np.median(angles))
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rw = int(abs(matrix[0, 0])*w+abs(matrix[0, 1])*h)
    rh = int(abs(matrix[0, 0])*h+abs(matrix[0, 1])*w)
    matrix[:, 2] += [(rw-w)/2, (rh-h)/2]
    rotated = cv2.warpAffine(img, matrix, (rw, rh), borderValue=(255, 255, 255))
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (rw, rh), flags=cv2.INTER_NEAREST)>0
    # Bring faint oblique lettering forward while suppressing dark body ink.
    g = rotated.astype(np.float32).mean(2)
    enhanced = 255-np.clip((255-g)*4,0,255)
    enhanced[g < 140] = 255
    observations = list(ocr.read_lines(rotated) or [])
    intact = sum(conf>=.65 and len(text)>=5 and ocr._geometry(box)[2]<6
                 and ocr._geometry(box)[1]>4*ocr._geometry(box)[0]
                 for box,text,conf in observations)
    sparse = intact < 4
    observations.extend(ocr.read_lines(np.repeat(enhanced.astype(np.uint8)[:,:,None],3,axis=2)) or [])
    seeds = []
    seed_texts = []
    for box, text, confidence in observations:
        height, length, tilt = ocr._geometry(box)
        if confidence >= (.45 if sparse else .65) and len(text) >= (3 if sparse else 5) and tilt < 6 and length > (2.0 if sparse else 4)*height:
            seeds.append((box, height, length))
            seed_texts.append(text)
    if not seeds:
        return None
    g = rotated.astype(np.float32).mean(2)
    size = max(7, int(np.median([s[1] for s in seeds])*.6)) | 1
    # Dark body type used to dominate the match score on dense paragraphs.
    # Estimate the repeated letter tone from OCR candidates and suppress darker
    # foreground for localization only; source pixels are never preprocessed.
    keepout = ocr.text_zones(img)
    rotated_keepout = np.zeros(g.shape,bool) if keepout is None else cv2.warpAffine(
        keepout.astype(np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    tones=[]
    neutral=rotated.max(2).astype(float)-rotated.min(2)<12
    for box,_,_ in seeds:
        lo=np.maximum(np.floor(box.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(box.max(0)).astype(int),[rw,rh])
        y,x=slice(lo[1],hi[1]),slice(lo[0],hi[0])
        values=g[y,x]
        selected=(values>50)&(values<235)&~rotated_keepout[y,x]&neutral[y,x]
        tones.append(values[selected].ravel())
    if sum(len(v) for v in tones)<50:
        return None
    tone=float(np.percentile(np.concatenate(tones),10))
    feature = np.minimum(255-g, 1.5*(255-tone)).astype(np.float32)
    feature[g < (.94 if sparse else .7)*tone] = 0
    if tone < 140:
        feature = np.minimum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((size,size),np.uint8)),150)
    options = []
    typical = float(np.median([s[2] for s in seeds]))
    for (box, height, length), token in zip(seeds,seed_texts):
        if not sparse and not .75*typical < length < 1.75*typical:
            continue
        x0,y0 = np.maximum(np.floor(box.min(0)).astype(int)-4, 0)
        x1,y1 = np.minimum(np.ceil(box.max(0)).astype(int)+4, [rw, rh])
        if y1-y0 < 8 or x1-x0 < 40:
            continue
        visible=valid[y0:y1,x0:x1]
        if visible.mean()<(.70 if sparse else .95) and not (sparse and np.count_nonzero(visible.mean(0)>.98)>=40):
            continue
        patch = feature[y0:y1, x0:x1]
        if patch.std() < 4:
            continue
        # Prefer a complete word away from underlying black text. OCR position
        # alone must never authorize erasing all pixels in its rectangle.
        black = float(np.mean(g[y0:y1, x0:x1] < 45))
        if not upright_body and black > .04:
            continue
        options.append((black, -length, x0, y0, x1, y1, token))
        # A boundary copy may expose a complete suffix while clipping its
        # prefix. Match the fully visible columns separately, without inventing
        # pixels outside the source page.
        if sparse and valid[y0:y1,x0:x1].mean()<.95:
            columns=np.flatnonzero(valid[y0:y1,x0:x1].mean(0)>.98)
            if columns.size>=40:
                lo,hi=x0+int(columns.min()),x0+int(columns.max())+1
                options.append((black-.001,-float(hi-lo),lo,y0,hi,y1,token+" visible"))
    chosen = None
    alternatives = []
    choices=[]
    seen=set()
    for option in sorted(options):
        token=''.join(c.lower() for c in option[-1] if c.isalnum())
        if sparse and token in seen:
            continue
        choices.append(option)
        seen.add(token)
    for _, _, x0,y0,x1,y1,_ in choices[:8]:
        patch = feature[y0:y1, x0:x1]
        th, tw = patch.shape
        scores = cv2.matchTemplate(feature, patch, cv2.TM_CCOEFF_NORMED)
        peaks = repeated._peaks(scores, patch.shape, .55, limit=256)
        if len(peaks) < 2:
            continue
        rank = len(peaks)*tw
        alternatives.append((rank,th,tw,peaks))
        if chosen is None or rank > chosen[0]:
            chosen = (rank, th, tw, peaks)
    if chosen is None:
        return None
    _, th, tw, peaks = chosen
    patches, luminance = [], []
    for x,y,_ in peaks:
        seen = valid[y:y+th,x:x+tw]
        patches.append(np.where(seen, feature[y:y+th,x:x+tw], np.nan))
        luminance.append(np.where(seen, g[y:y+th,x:x+tw], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        recurring = np.nan_to_num(np.nanpercentile(patches, 25 if tone < 140 and len(peaks)>=3 else 60, axis=0))
        light = np.nan_to_num(np.nanpercentile(luminance, 90, axis=0), nan=255)
    threshold = max(7., .18*float(np.percentile(recurring, 98)))
    template = (recurring > threshold).astype(np.uint8)
    if template.sum() < 50 or not .02 < template.mean() < .5:
        return None
    field = np.zeros((rh+2*th, rw+2*tw), np.float32)
    valid = np.pad(valid, ((th,th),(tw,tw)))
    feature = np.pad(feature, ((th,th),(tw,tw)))
    peaks = [(x+tw,y+th,s) for x,y,s in peaks]
    matrix[:,2] += [tw,th]
    partial = repeated._partial_matches(feature, valid, template, peaks)
    for x,y,_ in [*peaks,*partial]:
        np.maximum(field[y:y+th,x:x+tw], template, out=field[y:y+th,x:x+tw])
    # OCR can split a sparse watermark into a Chinese prefix and an ASCII
    # suffix. Keep separately confirmed word parts instead of erasing only the
    # most frequently matched part. Dense grids already supply enough samples.
    if sparse:
        extra=np.zeros((rh,rw),np.float32)
        for _,eh,ew,ep in sorted(alternatives,key=lambda a:a[0],reverse=True)[:6]:
            if sum(v>.65 for _,_,v in ep)<2:
                continue
            stack=[np.where(valid[y+th:y+th+eh,x+tw:x+tw+ew],
                     feature[y+th:y+th+eh,x+tw:x+tw+ew],np.nan) for x,y,_ in ep]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                recurring_part=np.nan_to_num(np.nanpercentile(stack,60,axis=0))
            part=(recurring_part>max(7.,.18*float(np.percentile(recurring_part,98)))).astype(np.float32)
            if not .02<part.mean()<.5:
                continue
            for x,y,_ in ep:
                np.maximum(extra[y:y+eh,x:x+ew],part,out=extra[y:y+eh,x:x+ew])
        np.maximum(field[th:th+rh,tw:tw+rw],extra,out=field[th:th+rh,tw:tw+rw])
    keepout = ocr.text_zones(img)
    protected = None if keepout is None else cv2.warpAffine(keepout.astype(np.uint8), matrix,
                    (valid.shape[1],valid.shape[0]),flags=cv2.INTER_NEAREST)>0
    for x,y,tx,ty,patch in repeated._edge_fragments(feature, valid, template, peaks, protected):
        ph,pw=patch.shape
        np.maximum(field[y:y+ph,x:x+pw], patch, out=field[y:y+ph,x:x+pw])
    field *= valid
    coverage = cv2.warpAffine(field, cv2.invertAffineTransform(matrix), (w,h))
    region = cv2.dilate((coverage>.12).astype(np.uint8), np.ones((3,3),np.uint8))>0
    if not .0001 < region.mean() < .15:
        return None
    samples = img[region & (gray>50)]
    if len(samples) < 50:
        return None
    # This path handles neutral lettering only. Coloured stamps have their own
    # measured matting path and must retain it.
    eligible = region & (gray>50) & (gray<245)
    if keepout is not None and np.count_nonzero(eligible & ~keepout)>50:
        eligible &= ~keepout
    bins = np.arange(0,258,2)
    histogram,_ = np.histogram(gray[eligible],bins=bins)
    if histogram.max()<20:
        return None
    level = float(bins[histogram.argmax()]+1)
    core = eligible & (np.abs(gray-level)<3)
    color = np.median(img[core],axis=0).astype(np.float32)
    if color.max()-color.min() > 12:
        return None
    paper = np.percentile(img.reshape(-1,3),97,axis=0).astype(np.float32)
    return MaskLayer(alpha=region.astype(np.float32),source=MaskSource.PRIOR,
        confidence=float(np.median([p[2] for p in peaks])),
        veil_strength=np.zeros(3,np.float32),veil_offset=color,
        paper=np.broadcast_to(paper,img.shape).copy(),footprint=region,
        method='single_repeated_paper',repair_mode='paper_repeat_cleanup',text_keepout=keepout)


def restore(img, mask):
    gray = img.astype(np.float32).mean(2)
    covered = float(mask.veil_offset.mean())
    # Real dark strokes are darker than the independently repeated paper glyph.
    # Include their antialiasing rim and coloured links/seals in the keep-out.
    ink = gray < max(35., .72*covered)
    if mask.text_keepout is not None:
        ink |= mask.text_keepout & (gray < .94*covered)
    ink |= img.max(2).astype(float)-img.min(2) > 18
    protected = cv2.dilate(ink.astype(np.uint8), np.ones((3,3),np.uint8))>0
    protected |= table_rules(img)
    region = mask.footprint & ~protected
    output = img.copy()
    output[region] = np.clip(mask.paper[region],0,255).astype(np.uint8)
    return output
