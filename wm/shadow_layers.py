"""Fit a coloured glyph plus its registered neutral shadow from current pixels.

No rendered glyph, transcript, filename, clean image or evaluation alpha is used.
Morphological operators are not used to create the selected pixels.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from dataclasses import replace


def prepare(image, legacy):
    """Default adapter for an already detected shadow-paper search domain.

    The source-only trial's fit now also rejects an opaque, saturated solution.
    Enabled at the user's request despite its recorded table/colour limitations;
    accepting a source fit does not certify restoration quality.
    """
    try:
        reconstructed, selected, observation = supported_inverse(image, legacy.footprint, .5)
        if (reconstructed.shape != image.shape or reconstructed.dtype != image.dtype
                or selected.shape != image.shape[:2] or selected.dtype != bool
                or not selected.any() or np.any(selected & ~legacy.footprint)
                or np.any(np.any(reconstructed != image, axis=2) & ~selected)):
            raise ValueError('阴影分层结果未通过尺寸或选区边界检查')
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return replace(legacy, shadow_evidence={
            'status': 'fallback', 'reason': str(exc), 'quality_verified': False})
    return replace(legacy, alpha=selected.astype(np.float32), footprint=selected,
                   method='single_observed_shadow_layers', repair_mode='shadow_layer_inverse',
                   shadow_reconstruction=reconstructed,
                   shadow_evidence={'status': 'accepted', 'observation': observation,
                                    'quality_verified': False,
                                    'known_limitations': '表格残影、彩色内容灰化及耗时退化尚未解决'})


def restore(image, mask, threshold=.08):
    """Write the recorded source reconstruction within the current permission.

    Editing may remove automatic strokes, so the original reconstruction must
    survive dataclass copies while the current selection controls every write.
    """
    reconstruction = mask.shadow_reconstruction
    if (reconstruction is None or reconstruction.shape != image.shape
            or reconstruction.dtype != image.dtype):
        raise ValueError('阴影分层观测缺失或与当前图片尺寸不一致')
    selected = mask.binary(threshold) > 0
    if mask.footprint is not None:
        selected &= mask.footprint
    output = image.copy()
    output[selected] = reconstruction[selected]
    return output


def supported_inverse(image, search, outlier_weight):
    gray = image.astype(np.float32).mean(2)
    chroma = image.max(2).astype(np.float32) - image.min(2)
    coloured = search & (chroma > 3)
    if coloured.sum() < 100:
        raise ValueError('源图颜色证据不足')
    ratio = float(np.percentile(gray[coloured]/chroma[coloured], 1))
    yy, xx = np.where(search)
    y0=max(0,int(yy.min())-30);y1=min(gray.shape[0],int(yy.max())+31)
    x0=max(0,int(xx.min())-30);x1=min(gray.shape[1],int(xx.max())+31)
    c = (chroma * search)[y0:y1,x0:x1]
    g = gray[y0:y1,x0:x1]
    domain = search[y0:y1,x0:x1]
    h,w=c.shape
    neutral = domain & (c<1) & (g>60)
    # Registration is verified against actual neutral deficits on current paper.
    # Clean white pixels also contribute, penalizing a shifted glyph in emptiness.
    ys,xs=np.where(neutral)
    ys=ys[::3];xs=xs[::3]
    deficit=(255-g[ys,xs])/255
    best=None
    for sigma in (1.,2.,3.,4.,5.):
        blurred=cv2.GaussianBlur(c,(0,0),sigma)
        for dy in range(-18,19,2):
            for dx in range(-18,19,2):
                values=cv2.remap(blurred,(xs-dx).astype(np.float32).reshape(-1,1),
                    (ys-dy).astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel()
                score=float(values@deficit/max(np.linalg.norm(values)*np.linalg.norm(deficit),1e-8))
                if best is None or score>best[0]:best=(score,dx,dy,sigma)
    correlation,dx,dy,sigma=best
    if correlation < .7:
        raise ValueError(f'当前图阴影配准未通过：{correlation:.3f}')
    # Use every third spatial observation, including both blank paper and body
    # intersections. A one-sided robust loss downweights dark body outliers;
    # paper observations must independently support the same compositing model.
    ys,xs=np.where(domain)
    ys=ys[::4];xs=xs[::4]
    cc=c[ys,xs]
    observed=(g[ys,xs]-ratio*cc)/255
    maxc=float(c.max())
    cache={}
    def shifted(sx,sy,blur):
        key=float(blur)
        if key not in cache:
            cache.clear();cache[key]=cv2.GaussianBlur(c,(0,0),key)
        return cv2.remap(cache[key],(xs-sx).astype(np.float32).reshape(-1,1),
                         (ys-sy).astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel()
    def residual(p):
        sx,sy,blur,a,b=p
        shadow=shifted(sx,sy,blur)
        transmission=(1-a*cc)*(1-b*shadow)
        error=transmission-observed
        return error*np.where(error>0,np.sqrt(outlier_weight),1.)
    init=[dx,dy,sigma,.85/maxc,.45/maxc]
    fit=least_squares(residual,init,bounds=([-22,-22,.4,.05/maxc,0],
                                          [22,22,8.,.99/maxc,.99/maxc]),
                      loss='soft_l1',f_scale=.01,diff_step=.01,max_nfev=120)
    # An opacity pinned to its upper bound cannot support a stable inverse of
    # covered ink. Use the optimiser's own constraint status, not sample rules.
    if not fit.success or fit.active_mask[3] == 1:
        raise ValueError('透叠模型未收敛或趋于不透明，保留原阴影清理')
    dx,dy,sigma,a,b=[float(v) for v in fit.x]
    blurred=cv2.GaussianBlur(c,(0,0),sigma)
    shadow=cv2.warpAffine(blurred,np.array([[1,0,dx],[0,1,dy]],np.float32),(w,h))
    t=(1-a*c)*(1-b*shadow)
    paper_pred=255*t+ratio*c
    errors=g-paper_pred
    paper_consistency=np.abs(errors)<5
    core_domain=domain&((c>1)|(shadow>.5))
    consistency=float(paper_consistency[core_domain].mean())
    if consistency < .55:
        raise ValueError(f'当前源图叠加关系验证不足：{consistency:.3f}')
    # Final neutral shadow selection requires registered glyph support AND an
    # actually nonwhite current input. The search box itself grants no pixels.
    selected=domain & (g<255) & ((c>.5)|((shadow>.1)&((255-paper_pred)>.5)))
    result=image.copy()
    bg=np.clip((g-ratio*c)/np.maximum(t,.06),0,255)
    predicted=np.repeat(bg[...,None],3,2).astype(np.uint8)
    target=result[y0:y1,x0:x1]
    target[selected]=predicted[selected]
    region=np.zeros(gray.shape,bool);region[y0:y1,x0:x1]=selected
    trace={'outlier_weight':outlier_weight,'registration_correlation':correlation,
           'measured_shift':[dx,dy],'measured_blur':sigma,
           'estimated_ink_to_chroma':ratio,'colour_opacity_scale':a,
           'shadow_opacity_scale':b,'paper_consistency_fraction':consistency,
           'paper_fit_error_percentiles':np.percentile(errors[core_domain],[5,25,50,75,95]).tolist(),
           'selected_pixels':int(region.sum()),'optimizer_success':bool(fit.success)}
    return result,region,trace
