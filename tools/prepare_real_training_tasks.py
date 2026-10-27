"""Prepare task-specific supervision for existing authorized real samples.

Does not run a removal model, generate a fictitious clean image, or overwrite
the strict paired manifest. Real scan boxes are approximate projected labels;
photographic before/after pairs retain their approximate-reference status.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data/watermark_training_increment_v1_1'
DEST = D / 'real_training'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path.relative_to(D).as_posix()


def box_from_mask(mask):
    yy, xx = np.where(mask > 0)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def row_base(source, category):
    return dict(id=source['id'], source_group=source['source_group'],
                split_group=source['split_group'], split='train',
                rights=source['rights'], training_admission=True,
                category=category, source_manifest=('weak_pairs.jsonl' if
                source.get('kind') == 'registered_real_image_pair' else 'references.jsonl'),
                paths={'input': source['paths']['watermarked']},
                clean_is_same_capture_ground_truth=False,
                annotation_input_is_not_required_at_product_inference=True)


def approximate_pair(source):
    paths = source['paths']
    clean = np.array(Image.open(D / paths['aligned_clean']).convert('RGB'))
    marked = np.array(Image.open(D / paths['watermarked']).convert('RGB'))
    valid = np.array(Image.open(D / paths['valid_area'])) > 0
    h, w = valid.shape
    ann = json.loads((D / paths['annotation']).read_text())
    boxes = ann['boxes']
    foreground_region = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        foreground_region[y1:y2, x1:x2] = 1
    # A smooth illumination-only adjustment estimated from light, unchanged
    # paper. No ink is synthesized or removed from the reference photograph.
    c = clean.astype(np.float32)
    m = marked.astype(np.float32)
    reliable_paper = (valid & (c.min(2) > 165) & (m.min(2) > 165)
                      & (np.max(np.abs(c - m), 2) < 20)
                      & (foreground_region == 0)).astype(np.float32)
    denominator = cv2.GaussianBlur(reliable_paper, (0, 0), 32)
    adjusted = np.empty_like(c)
    for channel in range(3):
        numerator = cv2.GaussianBlur((m[..., channel] - c[..., channel]) * reliable_paper,
                                   (0, 0), 32)
        offset = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                           where=denominator > .03)
        adjusted[..., channel] = c[..., channel] + np.clip(offset, -20, 20)
    adjusted = np.clip(adjusted, 0, 255).round().astype(np.uint8)
    delta = np.max(np.abs(adjusted.astype(np.int16) - marked.astype(np.int16)), 2)
    # Suppress unreliable differences outside known handwritten regions.
    # Xingtu has distributed white marks, so its common source support remains
    # valid and the entire pair receives a lower sampling/loss weight.
    reliable = valid.copy()
    if boxes:
        reliable &= (delta < 26) | (foreground_region > 0)
    reliable = cv2.erode(reliable.astype(np.uint8), np.ones((3, 3), np.uint8))
    folder = DEST / source['id']
    result = row_base(source, 'approximate_restoration_pair')
    result.update(tasks=['restoration'], size=[w, h], sample_weight=.25,
                  label_quality='registered_photographic_pair_with_residual_geometry_and_lighting_error',
                  mask_is_exact=False, alpha_available=False,
                  reference_derivation='existing before-photo homography plus smooth paper illumination adjustment; no inpainting',
                  intended_use='lower-weight restoration training, never exact pixel evaluation',
                  limitations=['paper deformation and subpixel registration remain',
                               'a valid reference region is not a watermark mask'])
    result['paths'].update(target=save_array(folder / 'target_approximate.png', adjusted),
                           loss_weight=save_array(folder / 'valid_loss_region.png', reliable * 255),
                           original_clean=paths['original_clean'],
                           original_registration=paths['registration'])
    result['qa'] = dict(registration=source['registration'],
                        valid_fraction=float(reliable.mean()),
                        unchanged_paper_median_before=float(np.median(np.abs(c-m)[reliable_paper > 0])),
                        unchanged_paper_median_after=float(np.median(np.abs(adjusted.astype(np.float32)-m)[reliable_paper > 0])))
    assert result['qa']['valid_fraction'] > .7
    save_json(folder / 'annotation.json', dict(boxes_xyxy=boxes, bbox_quality='coarse region only',
                                             training=result, verified_as_exact_pixel_pair=False))
    return result


def scan_detection(source, strict):
    pn = int(source['id'][-2:]) - 1
    with pymupdf.open(D / source['source_pdfs']['watermarked']) as doc:
        pix = doc[pn].get_pixmap(dpi=120, alpha=False)
        render = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    image = np.array(Image.open(D / source['paths']['watermarked']).convert('RGB'))
    h, w = image.shape[:2]
    sift = cv2.SIFT_create(nfeatures=6000)
    ks, ds = sift.detectAndCompute(cv2.cvtColor(render, cv2.COLOR_RGB2GRAY), None)
    kt, dt = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    matches = [a for a, b in cv2.BFMatcher().knnMatch(ds, dt, k=2) if a.distance < .7*b.distance]
    a = np.float32([ks[m.queryIdx].pt for m in matches])
    b = np.float32([kt[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(20261026)
    H, selected = cv2.findHomography(a, b, cv2.RANSAC, 3)
    assert H is not None
    error = np.linalg.norm(cv2.perspectiveTransform(a[:, None, :], H)[:, 0, :] - b, axis=1)
    inliers = selected.ravel() > 0
    assert int(inliers.sum()) >= 60 and np.median(error[inliers]) < 1.6
    style = source['watermark_subtype'].removeprefix('pdf_')
    doc_id = 'document_004' if 'D004' in source['id'] else 'document_006'
    record = next(r for r in strict if r.get('base_id') == doc_id and r.get('subtype') == style)
    assert record['split'] == 'train' and record['source_group'] == source['source_group']
    mask = np.array(Image.open(ROOT / 'data/watermark_training_v1' / record['pages'][pn]['paths']['mask']))
    mask = cv2.resize(mask, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST)
    warped = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)
    paper = cv2.warpPerspective(np.full(render.shape[:2], 255, np.uint8), H, (w, h))
    x1, y1, x2, y2 = box_from_mask(warped)
    # A padded object box is intentionally used instead of claiming precise
    # scan-space stroke segmentation under paper curvature and print blur.
    margin = 12
    bbox = [max(0, x1-margin), max(0, y1-margin), min(w, x2+margin), min(h, y2+margin)]
    result = row_base(source, 'paper_scan_box_detection')
    folder = DEST / source['id']
    result.update(tasks=['detection'], size=[w, h], sample_weight=.5,
                  label_quality='source_object_projected_box_visual_review_required',
                  boxes_xyxy=[bbox], mask_is_exact=False, alpha_available=False,
                  segmentation_available=False, restoration_target_available=False,
                  intended_use='bounding-box detector training only',
                  limitations=['projected box has print/curvature uncertainty',
                               'electronic clean PDF is not a same-capture restoration target'])
    result['paths'].update(projected_object_support=save_array(folder / 'projected_support_not_ground_truth.png', warped),
                           paper_support=save_array(folder / 'paper_support.png', paper),
                           source_clean_pdf=source['source_pdfs']['clean'])
    result['qa'] = dict(matches=len(matches), inliers=int(inliers.sum()),
                        median_error_px=float(np.median(error[inliers])),
                        p95_error_px=float(np.quantile(error[inliers], .95)), H=H.tolist(),
                        source_render_size=[render.shape[1], render.shape[0]], margin_px=margin)
    save_json(folder / 'annotation.json', result)
    return result


def red_detection(source):
    im = np.array(Image.open(D / source['paths']['watermarked']).convert('RGB'))
    r, g, b = (im[..., i].astype(np.int16) for i in range(3))
    mask = ((r-g > 28) & (r-b > 20) & (r > 65)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            keep[components == i] = 1
    assert .003 < keep.mean() < .12
    # Ignore uncertain color/antialias boundaries rather than labeling them
    # confidently as clean paper. These are pen labels, never a clean target.
    halo = cv2.dilate(keep, np.ones((5, 5), np.uint8)) > 0
    target = np.zeros_like(mask)
    target[halo] = 255
    target[keep > 0] = 1
    merged = cv2.dilate(keep, np.ones((19, 35), np.uint8))
    cnt, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    boxes = []
    for i in range(1, cnt):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        region = (labels == i) & (keep > 0)
        if region.any():
            boxes.append(box_from_mask(region))
    result = row_base(source, 'red_pen_segmentation')
    h, w = keep.shape
    result.update(tasks=['detection'], size=[w, h], sample_weight=.5,
                  label_quality='color_assisted_pen_annotation_visual_review_required',
                  boxes_xyxy=boxes, segmentation_available=True,
                  segmentation_encoding={'background': 0, 'watermark': 1, 'ignore': 255},
                  mask_is_exact=False, alpha_available=False, restoration_target_available=False,
                  intended_use='red-pen segmentation and box detection training only',
                  limitations=['no clean original; cannot supervise restoration',
                               'color-assisted labels; uncertain antialias boundary ignored'])
    result['paths']['segmentation'] = save_array(DEST / source['id'] / 'segmentation_labels.png', target)
    result['qa'] = dict(foreground_pixels=int(keep.sum()), ignored_pixels=int((target == 255).sum()),
                        boxes=len(boxes))
    save_json(DEST / source['id'] / 'annotation.json', result)
    return result


def preview(row):
    im = Image.open(D / row['paths']['input']).convert('RGB')
    draw = ImageDraw.Draw(im)
    for box in row.get('boxes_xyxy', []):
        draw.rectangle(box, outline=(240, 50, 20), width=4)
    if row.get('segmentation_available'):
        labels = np.array(Image.open(D / row['paths']['segmentation']))
        arr = np.array(im)
        arr[labels == 1] = (35, 175, 90)
        im = Image.fromarray(arr)
    im.thumbnail((600, 850))
    path = DEST / row['id'] / 'annotation_preview.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=88)
    return path


def main():
    original_manifest = digest(D / 'manifest.jsonl')
    strict = rows(ROOT / 'data/watermark_training_v1/manifest.jsonl')
    new = [approximate_pair(r) for r in rows(D / 'weak_pairs.jsonl')]
    for r in rows(D / 'references.jsonl'):
        new.append(scan_detection(r, strict) if r.get('task') == 'M03' else red_detection(r))
    for r in new:
        r['sha256'] = {key: digest(D / value) for key, value in r['paths'].items()}
    old = []
    for r in rows(D / 'manifest.jsonl'):
        old.append(dict(id=r['id'], split=r['split'], source_group=r['source_group'],
                        split_group=r['split_group'], rights=r['rights'], training_admission=True,
                        tasks=['restoration', 'detection'], category='strict_pair', size=r['size'],
                        sample_weight=1., label_quality=r['label_quality'],
                        paths=dict(input=r['paths']['watermarked'], target=r['paths']['clean'],
                                   mask=r['paths']['mask'], alpha=r['paths']['alpha']),
                        is_negative=r['is_negative'], strict_manifest='manifest.jsonl'))
    all_rows = old + new
    assert len(all_rows) == 129 and len({r['id'] for r in all_rows}) == 129
    (D / 'training_tasks.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in all_rows))
    save_json(DEST / 'manifest.json', dict(records=new, stage='prepared_pending_visual_review'))
    cards = []
    for r in new:
        p = preview(r)
        link = lambda p: quote('../' + p.relative_to(ROOT).as_posix(), safe='/')
        inputs = f'<a href="{link(D/r["paths"]["input"])}"><img src="{link(p)}"></a>'
        if 'target' in r['paths']:
            inputs += f'<a href="{link(D/r["paths"]["target"])}"><img src="{link(D/r["paths"]["target"])}"></a>'
        cards.append(f'<article id="{r["id"]}"><h3>{html.escape(r["id"])}</h3><p>{html.escape(r["category"])}</p><div class="pair">{inputs}</div><p>{html.escape(r["intended_use"])}</p></article>')
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实样本训练标注</title><style>body{font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;margin:0;background:#f1f4f8;color:#20324a}aside{position:fixed;inset:0 auto 0 0;width:190px;background:#172a43;color:white;padding:24px;overflow:auto}aside a{display:block;color:#d7e6fb;padding:8px;text-decoration:none}main{margin-left:240px;padding:28px;max-width:1120px}article,.note{background:white;padding:24px;border-radius:12px;margin-bottom:22px}.pair{display:flex;gap:20px}.pair a{width:50%}img{width:100%;max-height:790px;object-fit:contain}h1{font-size:30px}@media(max-width:700px){aside{position:static;width:auto}main{margin:0;padding:12px}.pair{display:block}.pair a{width:100%}}</style>'''
    nav = '<aside><b>真实样本训练标注</b><a href="#summary">状态与使用方法</a>'+''.join(f'<a href="#{r["id"]}">{html.escape(r["id"])}</a>' for r in new)+'</aside>'
    intro = '''<section class="note" id="summary"><h1>真实样本训练标注</h1><p>统一清单含 129 个训练样本：原 121 条严格配对 + 3 组低权重近似配对 + 5 张检测专用图片。修复训练可读取 124 条；检测训练可读取 126 条。两个任务重用原 121 条，数量不能相加成独立来源。</p><p>近似参考仍有光照、形变差异；扫描图先提供投影的目标框；红笔单图只提供检测标注。没有干净参考的图片不伪造修复答案。标注只用于训练，产品推理仅输入图片。</p><p>配对图：左为输入，右为近似干净参考。检测图：框表示检测标签；红笔图中的绿色表示前景标签。投影笔画图仅供核对，不作为精确分割真值。</p><pre>python tools/load_manual_increment.py --root data/watermark_training_increment_v1_1 --view training --task restoration\npython tools/load_manual_increment.py --root data/watermark_training_increment_v1_1 --view training --task detection</pre><p>本页展示数据标注，不包含去水印效果测试结果。</p></section>'''
    (ROOT / 'docs/真实样本训练标注.html').write_text(body+nav+'<main>'+intro+''.join(cards)+'</main></html>')
    assert digest(D / 'manifest.jsonl') == original_manifest
    print(json.dumps(dict(unique_training_records=len(all_rows), new_real_records=len(new),
                         restoration_records=sum('restoration' in r['tasks'] for r in all_rows),
                         detection_records=sum('detection' in r['tasks'] for r in all_rows),
                         stage='prepared_pending_visual_review'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
