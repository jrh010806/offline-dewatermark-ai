"""Locate repeated oblique lettering from observations in ONE input image.

OCR supplies direction and a seed line, not a hard-coded watermark phrase. A
second visual match is required. Registered repeats separate recurring strokes
from changing page content. No clean image, stored mask, or filename is read.
This path targets repeating diagonal text; other styles retain the tonal path.
"""
from __future__ import annotations
import warnings
import cv2
import numpy as np
from . import ocr
from .contract import MaskLayer, MaskSource


def _peaks(scores, shape, minimum, limit=12):
    scores = scores.copy()
    hh, ww = shape
    found = []
    for _ in range(limit):
        _, value, _, (x, y) = cv2.minMaxLoc(scores)
        if value < minimum or not np.isfinite(value):
            break
        found.append((x, y, value))
        scores[max(0,y-hh):y+hh, max(0,x-ww):x+ww] = -1
    return found


def _partial_matches(feature, valid, template, peaks):
    """Only accept clipped copies on rows supported by full visual repeats."""
    th, tw = template.shape
    ordered = np.array(sorted([p[:2] for p in peaks], key=lambda p:p[1]), np.float32)
    step = np.median(np.diff(ordered, axis=0), axis=0)
    period = float(np.linalg.norm(step))
    if period < th*2:
        return []
    normal = step / period
    phase = float(ordered[0] @ normal)
    _, cols = np.nonzero(template)
    if not cols.size:
        return []
    width = max(40, int(th*1.8))
    found = []
    for start in range(int(cols.min()), int(cols.max())-width+2, max(20,width//2)):
        part = template[:,start:start+width].astype(np.float32)
        if part.sum() < 30:
            continue
        score = cv2.matchTemplate(feature,part,cv2.TM_CCOEFF_NORMED)
        for x, y, confidence in _peaks(score,part.shape,.60,20):
            x -= start
            if x<0 or y<0 or x+tw>valid.shape[1] or y+th>valid.shape[0]:
                continue
            ratio = valid[y:y+th,x:x+tw][template>0].mean()
            row = (float(np.array([x,y])@normal)-phase)/period
            if not .02<ratio<.35 or abs(row-round(row))*period>th*.20:
                continue
            if any(np.linalg.norm(np.array([x,y])-p[:2])<th for p in [*peaks,*found]):
                continue
            found.append((x,y,confidence))
    return found


def _fragments(feature, template, peaks):
    """Locate short surviving pieces on the verified repeat rows.

    A watermark can be clipped before a document is rotated, so the fragment
    need not touch the final image border. Return only matched glyph pieces,
    never extrapolate an entire word across unrelated page content.
    """
    if len(peaks)<3:
        return []
    points=np.array(sorted([p[:2] for p in peaks],key=lambda p:p[1]),np.float32)
    step=np.median(np.diff(points,axis=0),axis=0)
    period=float(np.linalg.norm(step))
    th,tw=template.shape
    if period<th*2:
        return []
    normal=step/period
    phase=float(points[0]@normal)
    binary=(feature>30).astype(np.float32)
    width=max(35,int(th*1.2))
    found=[]
    for start in range(0,tw-width,max(16,width//2)):
        piece=template[:,start:start+width]
        yy,xx=np.nonzero(piece)
        if xx.size<40:
            continue
        x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
        patch=piece[y0:y1,x0:x1].astype(np.float32)
        score=cv2.matchTemplate(binary,patch,cv2.TM_CCOEFF_NORMED)
        for x,y,confidence in _peaks(score,patch.shape,.67,20):
            full=np.array([x-start-x0,y-y0])
            row=(float(full@normal)-phase)/period
            if abs(row-round(row))*period>th*.25:
                continue
            if any(np.linalg.norm(full-p[:2])<th for p in peaks):
                continue
            found.append((x,y,start+x0,y0,patch))
    return found


def _edge_fragments(feature, valid, template, peaks, protected=None):
    """Match surviving boundary strokes, without requiring a full glyph/word.

    The missing part is ignored in normalized correlation. Both visual shape
    and the already verified repeat rows must agree. Body-text zones veto the
    match; only matched visible strokes are returned, never a whole border.
    """
    if len(peaks) < 2:
        return []
    th, tw = template.shape
    points = np.array(sorted([p[:2] for p in peaks], key=lambda p: p[1]), np.float32)
    step = np.median(np.diff(points, axis=0), axis=0)
    period = float(np.linalg.norm(step))
    if period < th * 2:
        return []
    normal = step / period
    phase = float(points[0] @ normal)
    interior = cv2.erode(valid.astype(np.uint8), np.ones((th + 1, th + 1), np.uint8)) > 0
    band = valid & ~interior
    binary = (feature > 30).astype(np.uint8)
    candidates = cv2.morphologyEx(binary & band, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    pad = th
    padded = np.pad(template, ((pad, pad), (pad, pad))).astype(np.float32)
    found = []
    for x, y, w, h, area in stats[1:]:
        if area < 8 or h > th * 2 or w > th * 3:
            continue
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(valid.shape[1], x + w + 2), min(valid.shape[0], y + h + 2)
        query = binary[y0:y1, x0:x1].astype(np.float32)
        observed = valid[y0:y1, x0:x1].astype(np.uint8)
        count = float((query * observed).sum())
        if count < 8 or query.shape[0] >= padded.shape[0] or query.shape[1] >= padded.shape[1]:
            continue
        if protected is not None:
            overlap = (query * observed * protected[y0:y1, x0:x1]).sum() / count
            if overlap > .1:
                continue
        scores = cv2.matchTemplate(padded, query, cv2.TM_CCORR_NORMED, mask=observed)
        scores[~np.isfinite(scores)] = -1
        for _ in range(30):
            _, confidence, _, (tx, ty) = cv2.minMaxLoc(scores)
            if confidence < .67:
                break
            scores[max(0, ty - 2):ty + 3, max(0, tx - 2):tx + 3] = -1
            origin = np.array([x0 - tx + pad, y0 - ty + pad])
            row = (float(origin @ normal) - phase) / period
            if abs(row - round(row)) * period > th * .20:
                continue
            matched = padded[ty:ty + query.shape[0], tx:tx + query.shape[1]]
            agreement = float((matched * query * observed).sum()) / count
            if agreement < .65:
                continue
            found.append((x0, y0, tx - pad, ty - pad, matched * observed))
            break
    return found


def _document_overlay(img, region, profile, confidence, allow_broad_ink: bool = False,
                      coverage_hint: np.ndarray | None = None):
    """Fit covered-paper and covered-ink poles from THIS document only.

    The repeated shape supplies alpha at text intersections. Elsewhere white
    paper directly measures attenuation. Refuse the fit without both poles.
    """
    keepout=ocr.text_zones(img)
    if keepout is None or not keepout.any():
        return None
    gray=img.mean(2)
    paper=float(np.percentile(gray,97))
    estimated = np.clip(coverage_hint, 0, 1) if coverage_hint is not None else None
    edges=np.arange(0,258,2)
    histogram,_=np.histogram(gray[region & ~keepout],bins=edges)
    clean_peak = float(edges[int(np.argmax(histogram))])
    # Estimate the covered-paper pole on template-covered pixels only: the
    # whole region is mostly clean paper, so its near-white antialiasing can
    # otherwise outvote the actual veil pole (measured on layered samples).
    pool = gray[region & ~keepout & (estimated > .5)] if estimated is not None \
        else gray[region & ~keepout]
    if pool.size < 200:
        pool = gray[region & ~keepout]
    histogram,_=np.histogram(pool,bins=edges)
    histogram[(edges[:-1]<.16*paper)|(edges[:-1]>clean_peak-12)]=0
    if histogram.max()<20:
        return None
    covered=float(edges[np.argmax(histogram)]+1)
    if estimated is None:
        estimated = np.clip((paper-profile)/max(paper-covered,1),0,1)
    # Joint two-pole fit with a histogram peak for the covered-ink level.
    # The covered-ink pole sits between the uncovered body-ink level and the
    # covered-paper level; taking the strongest histogram peak in that open
    # interval avoids both pure-black contamination and antialiasing bias.
    body_pool = gray[keepout & ~region & (gray < .5 * paper)]
    pool = gray[region & keepout & (estimated > .5)]
    transmittance = ink = None
    covered_ink_level = None
    if body_pool.size >= 50 and pool.size >= 50:
        body_ink = float(np.percentile(body_pool, 20))
        hist, _ = np.histogram(pool, bins=np.arange(0, 258, 2))
        lo = int((body_ink + 4) // 2)
        hi = int(covered // 2)
        if hi > lo and hist[lo:hi].max() >= 10:
            covered_ink = float((int(np.argmax(hist[lo:hi])) + lo) * 2 + 1)
            if paper - body_ink > 30:
                s_fit = (covered - covered_ink) / (paper - body_ink)
                t_fit = covered - s_fit * paper
                if .08 < s_fit < .85 and 0 <= t_fit < covered:
                    transmittance, ink = float(s_fit), float(t_fit)
                    covered_ink_level = s_fit * body_ink + t_fit
    if transmittance is None:
        ink_pixels=gray[region & keepout & (estimated>.7) &
                        (gray>.05*paper) & (gray<.65*covered)]
        # The lower tail estimates black glyph cores, rather than the numerous
        # antialiased edges which would incorrectly raise the covered-ink pole.
        percentile = 10
        if ink_pixels.size < 20 and allow_broad_ink:
            broad = gray[region & keepout & (estimated > .5)]
            if broad.size < 50:
                return None
            ink_pixels = broad
            percentile = 5
        if ink_pixels.size < 20:
            return None
        ink = float(np.percentile(ink_pixels, percentile))
        transmittance = (covered - ink) / paper
        if not .08 < transmittance < .85:
            return None
    measured=np.clip((paper-gray)/max(paper-covered,1),0,1)
    # Undo the mild interpolation attenuation of the registered template; never
    # infer underlying glyph edges from their darkness alone. Inside the OCR
    # zones only the eroded high-confidence template core may drive the inverse:
    # antialiasing rims (clean or watermarked) are left untouched so a slightly
    # misregistered template cannot whiten or darken real body strokes.
    alpha=np.where(keepout,
                   np.where(estimated >= .15, measured, 0),
                   measured)
    if covered_ink_level is not None and transmittance < 0.4:
        # Heavy veils only: pixels darker than any veil-covered text can be
        # are clean body ink and must not enter the noisy inverse.
        alpha = np.where(keepout & (gray < covered_ink_level - 5), 0, alpha)
    alpha=np.where(region,alpha,0).astype(np.float32)
    return MaskLayer(alpha=alpha,source=MaskSource.PRIOR,confidence=confidence,
                     veil_strength=np.full(3,transmittance,np.float32),
                     veil_offset=np.full(3,ink,np.float32),
                     paper=np.full(img.shape,paper,np.float32),footprint=region,
                     repair_mode='document_overlay_full' if coverage_hint is not None else 'document_overlay',
                     method='repeated_document_overlay',
                     text_keepout=keepout)


def estimate(img: np.ndarray) -> MaskLayer | None:
    observations = ocr.read_lines(img)
    angles = []
    for box,text,confidence in observations or []:
        vector = box[1]-box[0]
        angle = float(np.degrees(np.arctan2(vector[1],vector[0])))
        height,length,_ = ocr._geometry(box)
        if confidence>=.65 and len(text)>=5 and length>3*height and 15<abs(angle)<60:
            angles.append(angle)
    if not angles:
        return None
    angle = float(np.median(angles))
    h,w = img.shape[:2]
    document=float((img.mean(2)>235).mean())>.65
    matrix = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    rw = int(abs(matrix[0,0])*w+abs(matrix[0,1])*h)
    rh = int(abs(matrix[0,0])*h+abs(matrix[0,1])*w)
    matrix[0,2] += (rw-w)/2
    matrix[1,2] += (rh-h)/2
    rotated = cv2.warpAffine(img,matrix,(rw,rh),borderValue=(255,255,255))
    valid = cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    seeds = []
    for box,text,confidence in ocr.read_lines(rotated) or []:
        height,length,tilt = ocr._geometry(box)
        if confidence>=.65 and len(text)>=5 and tilt<6 and length>4*height:
            seeds.append(box)
    if not seeds:
        return None
    box = max(seeds,key=lambda b:float(b[:,1].mean()))
    x0,y0 = np.floor(box.min(0)).astype(int)-3
    x1,y1 = np.ceil(box.max(0)).astype(int)+3
    x0=max(0,x0-int((y1-y0)*(.3 if document else 1.5)));x1=min(rw,x1+int((y1-y0)*.3))
    y0=max(0,y0);y1=min(rh,y1)
    th,tw=y1-y0,x1-x0
    if th<8 or tw<40 or tw>rw or th>rh:
        return None
    gray=rotated.mean(2).astype(np.uint8)
    size=max(5,int(ocr._geometry(box)[0]*.5)+1)|1
    feature=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((size,size),np.uint8)).astype(np.float32)
    feature=np.minimum(feature,150)
    suffix=x0+int(tw*.5)
    patch=feature[y0:y1,suffix:x1]
    if patch.std()<5:
        return None
    matches=_peaks(cv2.matchTemplate(feature,patch,cv2.TM_CCOEFF_NORMED),patch.shape,.50)
    peaks=[(x-(suffix-x0),y,s) for x,y,s in matches if x>=suffix-x0 and y+th<=rh and x-(suffix-x0)+tw<=rw]
    if len(peaks)<2:
        return None
    patches=[]
    luminance=[]
    for x,y,_ in peaks:
        patches.append(np.where(valid[y:y+th,x:x+tw],feature[y:y+th,x:x+tw],np.nan))
        luminance.append(np.where(valid[y:y+th,x:x+tw],gray[y:y+th,x:x+tw],np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)  # no observations outside rotated source
        recurring=np.nan_to_num(np.nanpercentile(patches,0 if len(peaks)<3 else 25,axis=0))
        light=np.nan_to_num(np.nanpercentile(luminance,90,axis=0),nan=255)
    template=(recurring>30).astype(np.uint8)
    if document:
        template[np.isfinite(patches).sum(axis=0)<2]=0
    if template.sum()<50:
        return None
    # Extra canvas permits partial instances whose full word extends off-image.
    feature=np.pad(feature,((th,th),(tw,tw)))
    valid=np.pad(valid,((th,th),(tw,tw)))
    peaks=[(x+tw,y+th,s) for x,y,s in peaks]
    matrix[:,2]+=[tw,th]
    partial=_partial_matches(feature,valid,template,peaks)
    field=np.zeros(valid.shape,np.float32)
    light_field=np.full(valid.shape,255,np.float32)
    for x,y,_ in [*peaks,*partial]:
        target=field[y:y+th,x:x+tw]
        np.maximum(target,template,out=target)
        light_field[y:y+th,x:x+tw]=np.minimum(light_field[y:y+th,x:x+tw],
                                             np.where(template>0,light,255))
    if document:
        for x,y,tx,ty,patch in _fragments(feature,template,peaks):
            ph,pw=patch.shape
            np.maximum(field[y:y+ph,x:x+pw],patch,out=field[y:y+ph,x:x+pw])
            values=np.where(patch>0,light[ty:ty+ph,tx:tx+pw],255)
            np.minimum(light_field[y:y+ph,x:x+pw],values,out=light_field[y:y+ph,x:x+pw])
        keepout = ocr.text_zones(img)
        protected = (cv2.warpAffine(keepout.astype(np.uint8), matrix,
                                   (valid.shape[1], valid.shape[0]),
                                   flags=cv2.INTER_NEAREST) > 0) if keepout is not None else None
        padded_light = np.pad(light, ((th, th), (th, th)), constant_values=255)
        for x, y, tx, ty, patch in _edge_fragments(feature, valid, template, peaks, protected):
            ph, pw = patch.shape
            target = field[y:y+ph, x:x+pw]
            np.maximum(target, patch, out=target)
            values = np.where(patch > 0, padded_light[ty+th:ty+th+ph, tx+th:tx+th+pw], 255)
            target_light = light_field[y:y+ph, x:x+pw]
            np.minimum(target_light, values, out=target_light)
    field*=valid
    coverage=cv2.warpAffine(field,cv2.invertAffineTransform(matrix),(w,h))
    region=cv2.dilate((coverage>.15).astype(np.uint8),np.ones((3,3),np.uint8))>0
    if not .0001<float(region.mean())<.20:
        return None
    confidence=float(min(p[2] for p in peaks))
    if document:
        profile=cv2.warpAffine(light_field,cv2.invertAffineTransform(matrix),(w,h),
                               borderValue=255)
        fitted=_document_overlay(img,region,profile,confidence,coverage_hint=coverage)
        if fitted is not None:
            return fitted
    # On white documents retain existing dark ink instead of asking a generic
    # inpainter to invent glyph strokes. Overlapping watermark may remain there.
    original_gray=img.mean(2)
    if document:
        inside=original_gray[(coverage>.5)&(original_gray>45)]
        if inside.size:
            paper_watermark=float(np.percentile(inside,25))
            region &= original_gray>max(35,min(65,paper_watermark*.65))
    color=np.median(img[region],axis=0).astype(np.float32)
    paper=np.broadcast_to(np.percentile(img.reshape(-1,3),97,axis=0).astype(np.float32),img.shape).copy()
    # Retain the registered glyph core separately from the padded repair
    # footprint. A matching template alone is insufficient at an individual
    # pixel: the original (unblurred) input must also show the dark stroke.
    native_feature = cv2.morphologyEx(original_gray.astype(np.float32),
                                     cv2.MORPH_BLACKHAT, np.ones((size,size),np.uint8))
    repeat_support = (coverage >= .75) & (native_feature > 30)
    return MaskLayer(alpha=region.astype(np.float32),source=MaskSource.PRIOR,
                     confidence=confidence,
                     veil_strength=np.zeros(3,np.float32),veil_offset=color,paper=paper,
                     footprint=region,repair_mode='inpaint',method='repeated_oblique_text',
                     repeat_stroke_support=repeat_support)
