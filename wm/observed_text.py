"""Complete observed watermark lettering without sample names or stored shapes.

Repeated photo type is confirmed by aligned OCR observations, then matched by
its observed colour-contrast shape. Shadow type requires oversized lettering
over a smaller printed body and a separate shadow beside its chromatic core.
"""
from __future__ import annotations
import cv2
import numpy as np
from . import ocr
from .contract import MaskLayer, MaskSource


def _layer(region, method, mode='inpaint', paper=None):
    return MaskLayer(region.astype(np.float32), footprint=region,
        confidence=.8, source=MaskSource.PRIOR, method=method, repair_mode=mode,
        veil_strength=np.ones(3,np.float32),veil_offset=np.zeros(3,np.float32),paper=paper)


def _bounded_repeat_footprint(strokes, matrix, size):
    """Reduce optional padding before discarding verified dense lettering.

    The area bound still applies to the observed strokes themselves. Never
    crop the mask spatially or raise the cap merely because it is dense.
    """
    inverse = cv2.invertAffineTransform(matrix)
    for kernel_size in (7, 5, 3, 1):
        padded = cv2.dilate(strokes, np.ones((kernel_size, kernel_size), np.uint8))
        footprint = cv2.warpAffine(padded, inverse, size, flags=cv2.INTER_NEAREST) > 0
        if .01 < footprint.mean() < .40:
            return footprint
    return None


def _protect_scene_text(footprint, source_lines):
    """Keep readable scene captions out of repeated oblique-watermark repair.

    Use only observations from the original image; enhanced watermark views
    cannot supply keep-out boxes. Intersections are left for manual review.
    """
    protected = np.zeros(footprint.shape, np.uint8)
    heights = [ocr._geometry(box)[0] for box, _, score in source_lines if score >= .5]
    if not heights:
        return footprint
    modal = float(np.median(heights))
    for box, text, score in source_lines:
        height, _, tilt = ocr._geometry(box)
        if score >= .5 and len(text.strip()) >= 2 and height <= 2.5*modal and tilt <= 15:
            cv2.fillConvexPoly(protected, np.rint(box).astype(np.int32), 1)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8)) > 0
    return footprint & ~protected


def _positive_ink_view(feature, observations):
    """Reject enhanced views whose bright foreground is mostly word background."""
    fractions = []
    for box, _, _ in observations:
        domain = np.zeros(feature.shape, np.uint8)
        cv2.fillConvexPoly(domain, np.rint(box).astype(np.int32), 1)
        if domain.any():
            fractions.append(float(np.mean(feature[domain > 0] > 3)))
    return bool(fractions) and float(np.median(fractions)) <= .50


def _stroke_template(typical):
    """Hysteresis on repeated pixel evidence; holes and whitespace remain zero."""
    high, _ = cv2.threshold(np.uint8(np.clip(typical, 0, 12)*20), 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high = max(2.5, min(6., high/20))
    weak = typical > max(1.2, high*.45)
    count, labels, _, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[typical > high])] = True
    keep[0] = False
    return keep[labels]


def repeated_photo(image, source_lines, *, stroke_only=False, search_mask=None):
    h,w=image.shape[:2]
    candidates=[]
    for box,text,score in source_lines:
        height,length,tilt=ocr._geometry(box)
        if score>=.65 and len(text.strip())>=4 and .025*min(h,w)<=height<=.10*min(h,w) and length>2.5*height and 15<tilt<55:
            candidates.append((box,float(height),float(tilt)))
    # Two separated observations may seed colour enhancement. Acceptance still
    # requires at least three enhanced observations and three visual matches.
    if len(candidates)<2:return None
    height=float(np.median([a[1] for a in candidates]));tilt=float(np.median([a[2] for a in candidates]))
    candidates=[a for a in candidates if .65*height<a[1]<1.5*height and abs(a[2]-tilt)<6]
    if len(candidates)<2:return None
    centres=np.array([a[0].mean(0) for a in candidates])
    if np.linalg.norm(np.ptp(centres, axis=0)) < 2*height:return None
    rgb=image.astype(np.float32);choices=[]
    for channel in range(3):
        opponent=np.uint8(np.clip(128+rgb[...,channel]-(rgb[...,(channel+1)%3]+rgb[...,(channel+2)%3])/2,0,255))
        feature=cv2.morphologyEx(opponent,cv2.MORPH_TOPHAT,np.ones((17,17),np.uint8)).astype(np.float32)
        view=np.uint8(255-np.clip(feature*12,0,255))
        observations=[]
        for box,text,score in ocr.read_lines(np.repeat(view[...,None],3,axis=2)) or []:
            bh,bl,bt=ocr._geometry(box)
            if score>=.65 and len(text.strip())>=3 and .6*height<bh<1.6*height and bl>2*bh and abs(bt-tilt)<7:
                observations.append((box,float(bh),float(score)))
        if len(observations)>=3:
            centres = np.array([a[0].mean(0) for a in observations])
            if np.ptp(centres[:,0]) >= 3*height and np.ptp(centres[:,1]) >= 2*height:
                if stroke_only:
                    # OCR can read light holes on a dark enhanced view. Such
                    # a channel highlights the background around the letters,
                    # producing a solid block if mistaken for the ink stencil.
                    if not _positive_ink_view(feature, observations):
                        continue
                choices.append((len(observations),feature,observations,channel))
    if not choices:return None
    _,feature,observations,chosen_channel=max(choices,key=lambda a:a[0])
    vectors=[a[0][1]-a[0][0] for a in candidates]
    angle=float(np.median([np.degrees(np.arctan2(v[1],v[0])) for v in vectors]))
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    rw=int(abs(matrix[0,0])*w+abs(matrix[0,1])*h)
    rh=int(abs(matrix[0,0])*h+abs(matrix[0,1])*w)
    matrix[:,2]+=[(rw-w)/2,(rh-h)/2]
    field=cv2.warpAffine(feature,matrix,(rw,rh))
    valid=cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    signal=np.clip(field/8,0,1).astype(np.float32)
    if stroke_only:
        opponent = np.uint8(np.clip(128+rgb[...,chosen_channel]-(rgb[...,(chosen_channel+1)%3]+rgb[...,(chosen_channel+2)%3])/2,0,255))
        size = min(31, max(9, int(.45*height) | 1))
        both_signs = np.abs(opponent.astype(np.float32)-cv2.medianBlur(opponent, size))
        both_signs = cv2.GaussianBlur(both_signs, (0, 0), .5)
        matching_signal = np.clip(cv2.warpAffine(both_signs,matrix,(rw,rh))/8,0,1).astype(np.float32)
        if search_mask is not None:
            observed = cv2.warpAffine(search_mask.astype(np.uint8), matrix, (rw,rh), flags=cv2.INTER_NEAREST)
            active = (observed.sum(1) > max(5, .025*rw)).astype(np.uint8)
            active = cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,
                                      np.ones((max(3,int(.25*height)),1),np.uint8))[:,0]
            row_search = np.broadcast_to(active[:,None] > 0,(rh,rw))
        else:
            row_search = None
    options=[];domains=np.zeros((rh,rw),np.uint8)
    for box,bh,score in observations:
        b=cv2.transform(box[None].astype(np.float32),matrix)[0]
        cv2.fillConvexPoly(domains,np.rint(b).astype(np.int32),1)
        x0,y0=np.maximum(np.floor(b.min(0)).astype(int)-3,0)
        x1,y1=np.minimum(np.ceil(b.max(0)).astype(int)+4,[rw,rh])
        width=min(x1-x0,max(24 if stroke_only else 40,int((1.6 if stroke_only else 5)*height)))
        for x in range(x0,max(x0+1,x1-width+1),max(12 if stroke_only else 20,int((.8 if stroke_only else 3)*height))):
            patch=signal[y0:y1,x:x+width]
            support=(field[y0:y1,x:x+width]>3)&valid[y0:y1,x:x+width]
            if patch.shape[0]<12 or patch.shape[1]<40 or not .05<support.mean()<.4:continue
            # Prefer uncluttered source observations: type has blank spaces.
            options.append((float(np.mean(patch<.15)),patch.copy(),support.copy(),x,y0))
    if not options:return None
    result=np.zeros((rh,rw),np.uint8)
    # Deskew the verified direction before another OCR observation. Oblique
    # full-frame OCR joins adjacent words or misses them over dark backgrounds.
    # After deskew only horizontal, same-size observations can extend the lines.
    upright=np.uint8(255-np.clip(field*12,0,255))
    for box,text,score in ocr.read_lines(np.repeat(upright[...,None],3,axis=2)) or []:
        bh,bl,bt=ocr._geometry(box)
        if score<.60 or len(text.strip())<3 or bt>8 or not .55*height<bh<1.65*height or bl<2*bh:continue
        cv2.fillConvexPoly(domains,np.rint(box).astype(np.int32),1)
    domains=cv2.dilate(domains,np.ones((5,5),np.uint8))
    if not stroke_only:
        result[(domains>0)&valid&(field>2)]=1
    matches=0
    for _,patch,support,source_x,source_y in sorted(options,key=lambda a:a[0],reverse=True)[:24 if stroke_only else 8]:
        ph,pw=patch.shape
        scores=cv2.matchTemplate(signal,patch,cv2.TM_CCOEFF_NORMED)
        found=[]
        for _ in range(100):
            _,value,_,(x,y)=cv2.minMaxLoc(scores)
            if not np.isfinite(value) or value<(.62 if stroke_only else .58):break
            found.append((x,y));scores[max(0,y-ph//2):y+ph//2+1,max(0,x-pw//2):x+pw//2+1]=-1
        if len(found)<3:continue
        stack=np.stack([field[y:y+ph,x:x+pw] for x,y in found])
        typical = np.percentile(stack,50 if stroke_only else 30,axis=0)
        if stroke_only:
            recurring = _stroke_template(typical)
        else:
            recurring=typical>1
        if not .04<recurring.mean()<.45:continue
        if stroke_only:
            # Learn ink only from positive-colour copies, then match the same
            # validated glyph through backgrounds where its contrast reverses.
            match_patch = matching_signal[source_y:source_y+ph,source_x:source_x+pw]
            extra_scores = cv2.matchTemplate(matching_signal,match_patch,cv2.TM_CCOEFF_NORMED)
            for _ in range(100):
                _, value, _, (x,y) = cv2.minMaxLoc(extra_scores)
                if not np.isfinite(value) or value < (.65 if row_search is not None else .72):
                    break
                extra_scores[max(0,y-ph//2):y+ph//2+1,max(0,x-pw//2):x+pw//2+1] = -1
                local_signal = matching_signal[y:y+ph,x:x+pw]
                ink_contrast = float(local_signal[recurring].mean()-local_signal[~recurring].mean())
                if ink_contrast >= .08 and (row_search is None or row_search[min(rh-1,y+ph//2), min(rw-1,x+pw//2)]):
                    found.append((x,y))
        for x,y in found:
            stencil = (recurring & (matching_signal[y:y+ph,x:x+pw] > .12)) if stroke_only else recurring
            result[y:y+ph,x:x+pw]|=stencil.astype(np.uint8)
        if stroke_only:
            # A full learned glyph stencil may continue through the frame edge.
            # Border copies supply no zeros to the shape learned above.
            padded_signal = cv2.copyMakeBorder(signal, ph, ph, pw, pw, cv2.BORDER_CONSTANT)
            edge_scores = cv2.matchTemplate(padded_signal, patch, cv2.TM_CCOEFF_NORMED)
            edge_scores[ph:signal.shape[0]+1, pw:signal.shape[1]+1] = -1
            for _ in range(60):
                _, value, _, (ex, ey) = cv2.minMaxLoc(edge_scores)
                if not np.isfinite(value) or value < .65:
                    break
                edge_scores[max(0, ey-ph//2):ey+ph//2+1, max(0, ex-pw//2):ex+pw//2+1] = -1
                x, y = ex-pw, ey-ph
                left, right = max(0, x), min(signal.shape[1], x+pw)
                top, bottom = max(0, y), min(signal.shape[0], y+ph)
                if right-left >= .35*pw and bottom-top >= .35*ph:
                    stencil = recurring[top-y:bottom-y, left-x:right-x] & (matching_signal[top:bottom,left:right] > .12)
                    result[top:bottom, left:right] |= stencil.astype(np.uint8)
        matches+=len(found)
    if matches<3:return None
    footprint = (cv2.warpAffine(result, cv2.invertAffineTransform(matrix), (w, h),
                               flags=cv2.INTER_NEAREST) > 0) if stroke_only else _bounded_repeat_footprint(result, matrix, (w, h))
    if footprint is None:return None
    if not stroke_only:
        from . import photo_repeat_inverse
        inverse = photo_repeat_inverse.estimate(image, footprint)
        if inverse is not None:
            return inverse
    if not stroke_only:
        footprint = _protect_scene_text(footprint, source_lines)
    if footprint.mean() <= .01:return None
    return _layer(footprint,'recall_repeated_strokes' if stroke_only else 'photo_observed_repeats')


def shadow_paper(image, source_lines):
    h,w=image.shape[:2];rgb=image.astype(np.float32);gray=rgb.mean(2)
    if np.mean(gray>235)<.65:return None
    body=[(box,float(ocr._geometry(box)[0])) for box,text,score in source_lines
          if score>=.65 and len(text)>=5 and ocr._geometry(box)[2]<8 and ocr._geometry(box)[0]<.035*min(h,w)]
    if len(body)<5:return None
    body_height=float(np.median([a[1] for a in body]))
    chroma=rgb.max(2)-rgb.min(2)
    view=np.uint8(255-np.clip(chroma*12,0,255))
    candidates=[]
    for box,text,score in ocr.read_lines(np.repeat(view[...,None],3,axis=2)) or []:
        bh,bl,bt=ocr._geometry(box);cy=box[:,1].mean()
        if score<.7 or len(text)<4 or bh<max(2.5*body_height,.045*min(h,w)) or bl<3*bh or bt>8 or not .18*h<cy<.85*h:continue
        if not any(b[:,1].mean()<box[:,1].min() for b,_ in body) or not any(b[:,1].mean()>box[:,1].max() for b,_ in body):continue
        candidates.append((box,float(bh)))
    region=np.zeros((h,w),bool)
    for box,bh in candidates:
        domain=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(domain,np.rint(box).astype(np.int32),1)
        core=(domain>0)&(chroma>3)
        if not .07<core.sum()/max(domain.sum(),1)<.65:continue
        radius=max(3,int(.15*bh))
        near=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
        # A separate neutral halo is shadow evidence, not permission to erase
        # an ordinary large coloured heading elsewhere in a document.
        shadow=near&~core&(chroma<3)&(gray<225)&(gray>90)
        if shadow.sum()<.05*core.sum():continue
        # A drop shadow extends beyond the coloured core. The independently
        # observed line bounds this local search; include its neutral outline.
        pad=max(3,int(.2*bh))
        region|=cv2.dilate(domain,np.ones((2*pad+1,2*pad+1),np.uint8))>0
    if not region.any() or region.mean()>.12:return None
    return _layer(region,'single_observed_shadow','shadow_paper_cleanup',np.full(image.shape,255,np.float32))


def restore_shadow(image,mask):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    size=7
    background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
    # Subtract the observed covered-paper profile rather than copying coloured
    # stroke fragments. Small printed rows define where ink may be retained.
    lines=ocr.read_lines(image) or []
    body=[]
    for box,text,score in lines:
        bh,bl,bt=ocr._geometry(box)
        if score>=.6 and len(text)>=3 and bt<8 and 7<bh<.035*min(gray.shape):body.append((box,float(bh)))
    keep=np.zeros(gray.shape,np.uint8)
    for box,bh in body:
        cv2.fillConvexPoly(keep,np.rint(box).astype(np.int32),1)
    difference=np.maximum(0,background.astype(float)-gray)
    contrast=difference/np.maximum(background,40)
    ink=(keep>0)&(contrast>.25)
    value=np.uint8(np.clip(255*(1-np.clip(contrast*1.8,0,1)),0,255))
    out=image.copy();out[mask.footprint]=255
    out[mask.footprint&ink]=np.repeat(value[...,None],3,axis=2)[mask.footprint&ink]
    return out
