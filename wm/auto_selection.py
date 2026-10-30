"""Experimental source-only comparison of existing repair candidates.

Search domains and pattern observations never become repair permission. These
signals can rank candidates or flag risk; they cannot certify watermark removal
or reconstruct text that was unreadable in the source. No references enter here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import re
from time import perf_counter

import cv2
import numpy as np

from . import ocr, recall
from .contract import FailureCode

MIN_IMPROVEMENT = .10
SOURCE_CONFIDENCE = .85
PERSISTENCE_WARNING = .35


@dataclass
class Evidence:
    patterns: list
    readable: list
    lines: list
    ocr_available: bool


def _same_repair(a, b):
    if a.method != b.method or a.repair_mode != b.repair_mode or a.confidence != b.confidence:
        return False
    for name in ('alpha', 'footprint', 'text_keepout', 'overlay_opacity',
                 'veil_strength', 'veil_offset', 'paper', 'selection'):
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            if x is not y:
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def _polygon(shape, box):
    domain = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(domain, np.rint(box).astype(np.int32), 1)
    return domain.astype(bool)


def _feature(image, channel, operation, size):
    rgb = image.astype(np.float32)
    view = (rgb.mean(2) if channel == 0 else
            rgb[..., channel-1] - (rgb.sum(2)-rgb[..., channel-1])/2)
    return cv2.morphologyEx(view, operation, np.ones((size, size), np.uint8))


def observe(image, candidates, threshold, *, use_ocr=True):
    """Freeze common observations before seeing any candidate output."""
    raw = ocr.read_lines(image) if use_ocr else None
    observations = recall._lines(image, None) if raw is not None else []
    # Union is a search boundary only. All scoring pixels also require native
    # source contrast; no union is passed to an inpainter or stored as a mask.
    search = np.zeros(image.shape[:2], bool)
    for mask in candidates:
        search |= mask.binary(threshold) > 0
    patterns = []
    seen = []
    for box, _, _ in observations:
        height, _, _ = ocr._geometry(box)
        if any(np.linalg.norm(box.mean(0)-centre) < height for centre in seen):
            continue
        domain = _polygon(search.shape, box)
        selected = domain & search
        if selected.sum() < 20:
            continue
        size = min(51, max(9, int(height*.5) | 1))
        best = None
        for channel in range(4):
            for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
                feature = _feature(image, channel, operation, size)
                control = feature[domain & ~search]
                noise = float(np.percentile(control, 90)) if control.size else 0.
                supported = selected & (feature > max(6., noise+3.))
                if supported.sum() >= 20:
                    # Compare the strongest independently visible pattern for
                    # this observation, not whichever candidate selected more.
                    strength = float(np.percentile(feature[supported], 70))
                    if best is None or strength > best[0]:
                        best = strength, supported, channel, operation, size, feature
        if best is not None:
            _, support, channel, operation, size, feature = best
            y, x = np.nonzero(support)
            patterns.append((y, x, channel, operation, size, feature[y, x]))
            seen.append(box.mean(0))
    readable = []
    for box, text, confidence in raw or []:
        height, _, tilt = ocr._geometry(box)
        token = _token(text)
        if confidence >= SOURCE_CONFIDENCE and tilt <= 8 and token:
            # Even text intersecting candidate masks is checked. Excluding it
            # would let an oversized mask hide the very content it destroys.
            readable.append((box, token, float(height)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    length = max(30, int(min(gray.shape)*.08))
    segments = cv2.HoughLinesP(edges, 1, np.pi/180, max(20, length//2),
                              minLineLength=length, maxLineGap=3)
    lines = []
    for x0, y0, x1, y1 in (segments.reshape(-1, 4) if segments is not None else []):
        angle = abs(np.degrees(np.arctan2(y1-y0, x1-x0))) % 180
        if min(angle, abs(angle-90), abs(angle-180)) > 5:
            continue
        domain = np.zeros(gray.shape, np.uint8)
        cv2.line(domain, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
        y, x = np.nonzero((domain > 0) & (edges > 0))
        if len(y) >= length*.5:
            lines.append((y, x))
    return Evidence(patterns, readable, lines, raw is not None)


def _token(text):
    return re.sub(r'\s+', '', text).casefold()


def _overlap(a, b):
    # Axis-aligned overlap only associates horizontal source body text. It is
    # never used as evidence that all pixels in either box are watermarks.
    amin, amax, bmin, bmax = a.min(0), a.max(0), b.min(0), b.max(0)
    intersection = np.maximum(0, np.minimum(amax, bmax)-np.maximum(amin, bmin)).prod()
    return float(intersection/max(1., (amax-amin).prod()))


def assess(source, result, evidence, thr):
    output = result.output.image
    if output.shape != source.shape or output.dtype != np.uint8:
        raise ValueError('候选输出尺寸或像素类型改变')
    selected = result.mask.binary(thr.mask_alpha_min) > 0
    permission = selected.copy()
    if thr.lama_mask_dilate > 0:
        size = 2*int(thr.lama_mask_dilate)+1
        permission = cv2.dilate(permission.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size))) > 0
    changed = np.any(output != source, axis=2)
    outside = int(np.count_nonzero(changed & ~permission))
    patterns = []
    cache = {}
    for y, x, channel, operation, size, strength in evidence.patterns:
        key = channel, operation, size
        if key not in cache:
            cache[key] = _feature(output, *key)
        observed = cache[key][y, x]
        patterns.append(float(np.minimum(observed/np.maximum(strength, 1), 1).mean()))
    # OCR can be unavailable independently on an output; do not turn unknown
    # into a zero-loss score or use the other candidate's larger box as a pass.
    lines = ocr.read_lines(output) if evidence.ocr_available else None
    lost = None
    if lines is not None:
        lost = sum(not any(_token(text) == token and confidence >= .45
                          and _overlap(box, found) >= .25
                          for found, text, confidence in lines)
                   for box, token, _ in evidence.readable)
    edges = cv2.Canny(cv2.cvtColor(output, cv2.COLOR_RGB2GRAY), 60, 140)
    near = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
    structural_loss = (float(np.mean([1-float(near[y, x].mean())
                                     for y, x in evidence.lines])) if evidence.lines else None)
    return dict(outside_permission_pixels=outside,
                readable_source_items=len(evidence.readable), missing_readable_items=lost,
                observed_pattern_persistence=float(np.mean(patterns)) if patterns else None,
                observed_patterns=len(patterns), observed_pattern_pixels=sum(len(p[0]) for p in evidence.patterns),
                source_line_loss=structural_loss, source_lines=len(evidence.lines),
                selected_pixels=int(selected.sum()), changed_pixels=int(changed.sum()),
                note='原图观测的风险信号；不是真实水印残留率、正文正确率或成功率')


def choose(records, preferred):
    """Switch only on a common, material improvement without added content risk."""
    eligible = [r for r in records if r.get('status') == 'completed'
                and r['assessment']['outside_permission_pixels'] == 0]
    if not eligible:
        raise RuntimeError('所有候选均失败或越过修复许可，未交付越界结果')
    selected = next((r for r in eligible if r['name'] == preferred), eligible[0])
    reason = '证据不足或差异不明确，保留原处理模式'
    for candidate in eligible:
        if candidate is selected:
            continue
        a, b = candidate['assessment'], selected['assessment']
        if a['missing_readable_items'] is None or b['missing_readable_items'] is None:
            continue
        content_ok = (a['missing_readable_items'] <= b['missing_readable_items']
                      and (a['source_line_loss'] is not None and b['source_line_loss'] is not None
                           and a['source_line_loss'] <= b['source_line_loss']+.01
                           or a['source_line_loss'] is None and b['source_line_loss'] is None))
        pa, pb = a['observed_pattern_persistence'], b['observed_pattern_persistence']
        if content_ok and pa is not None and pb is not None and pb-pa >= MIN_IMPROVEMENT:
            selected = candidate
            reason = '同一原图笔画痕迹明显减少，且未新增已观察到的文字或线条损失'
        elif (a['missing_readable_items'] < b['missing_readable_items']
              and (pa is None and pb is None or pa is not None and pb is not None and pa <= pb+.01)
              and content_ok):
            selected = candidate
            reason = '保留了更多原本可读文字，且原图笔画痕迹未明显增加'
    if selected['name'] != preferred and not any(r['name'] == preferred for r in eligible):
        reason = '原模式候选失败或越界，展示仍可用的候选并标记待复核'
    return selected['name'], reason


def preselect(candidates, evidence, threshold, preferred):
    """Rank source-supported strokes before any expensive repair.

    These are incomplete source observations, never watermark truth or new
    mask pixels. Unknown observations preserve the existing mode.
    """
    scores = {}
    for name, mask in candidates:
        selected = mask.binary(threshold) > 0
        coverage = [float(selected[y, x].mean()) for y, x, *_ in evidence.patterns]
        text = [float(selected[_polygon(selected.shape, box)].mean())
                for box, _, _ in evidence.readable
                if _polygon(selected.shape, box).any()]
        lines = [float(selected[y, x].mean()) for y, x in evidence.lines]
        scores[name] = dict(source_pattern_coverage=float(np.mean(coverage)) if coverage else None,
                            readable_overlap=float(np.mean(text)) if text else None,
                            line_overlap=float(np.mean(lines)) if lines else None)
    chosen = preferred
    reason = '原图依据不足或差异不明确，先用原处理模式'
    for name, _ in candidates:
        if name == chosen:
            continue
        a, b = scores[name], scores[chosen]
        ca, cb = a['source_pattern_coverage'], b['source_pattern_coverage']
        safe = all(a[k] is None and b[k] is None or
                   a[k] is not None and b[k] is not None and a[k] <= b[k]+tolerance
                   for k, tolerance in (('readable_overlap', .02), ('line_overlap', .05)))
        if safe and ca is not None and cb is not None and ca-cb >= MIN_IMPROVEMENT:
            chosen = name
            reason = '修复前发现该方法覆盖更多原图可见笔画，未明显增加已观察到的文字或直线交叠'
    return chosen, reason, scores


def retry_reason(record, current, alternative):
    """Spend a second repair only when it has a concrete purpose."""
    if record['status'] == 'error':
        return '首次修复运行失败，尝试另一方法'
    a = record['assessment']
    if a['outside_permission_pixels']:
        return '首次输出越过修改许可，尝试另一方法'
    safe = all(alternative[k] is None and current[k] is None or
               alternative[k] is not None and current[k] is not None and
               alternative[k] <= current[k]+tolerance
               for k, tolerance in (('readable_overlap', .02), ('line_overlap', .05)))
    ca, cb = alternative['source_pattern_coverage'], current['source_pattern_coverage']
    persistence = a['observed_pattern_persistence']
    if (safe and ca is not None and cb is not None and ca-cb >= MIN_IMPROVEMENT
            and persistence is not None and persistence >= .80):
        return '首次输出大部分共同痕迹未消失，另一方法有更充分的原图笔画依据'
    if (a['missing_readable_items'] not in (None, 0) and ca is not None and cb is not None
            and ca >= cb-.01 and current['readable_overlap'] is not None
            and alternative['readable_overlap'] is not None
            and current['readable_overlap']-alternative['readable_overlap'] >= .02 and safe):
        return '首次输出丢失原本可读文字，另一方法更少覆盖这些文字'
    return None


def process(page, thr, *, prior=None, close_px=11, use_ocr=True, style=None,
            mask_ms=0, checkpoint=None):
    """Preselect from source observations; repair again only for a stated need."""
    from . import masks, pipeline
    start = perf_counter()
    base_thr = replace(thr, watermark_recall_priority=False)
    base = masks.estimate_single_page(page.image, base_thr, close_px=close_px,
                                      use_ocr=use_ocr, style=style)
    enhanced = recall.expand(page.image, deepcopy(base), thr, use_ocr=use_ocr, style=style)
    enhanced.instances = masks._instances(enhanced.alpha, thr)
    candidates = [('original', masks.with_prior(deepcopy(base), prior, base_thr)),
                  ('stroke_completion', masks.with_prior(enhanced, prior, thr))]
    identical = _same_repair(candidates[0][1], candidates[1][1])
    evidence = (Evidence([], [], [], False) if identical else
                observe(page.image, [m for _, m in candidates], thr.mask_alpha_min, use_ocr=use_ocr))
    preferred = 'stroke_completion' if thr.watermark_recall_priority else 'original'
    initial, initial_reason, source_scores = preselect(candidates, evidence, thr.mask_alpha_min, preferred)
    candidates.sort(key=lambda item: item[0] != initial)
    detection_ms = int((perf_counter()-start)*1000)+mask_ms
    records, results = [], {}
    repair_count, fallback_reason = 0, None
    for name, mask in candidates:
        if checkpoint is not None and not checkpoint():
            raise InterruptedError('自动比较已停止，保留之前结果')
        try:
            # Deep copies isolate mutable confidence/physical fields and the
            # batch's cached prior from failed or rejected candidates.
            if identical and results:
                first = next(iter(results.values()))
                result = replace(first, processing=dict(first.processing))
                assessment = dict(records[0]['assessment'])
            else:
                repair_count += 1
                result = pipeline._process_one(page, deepcopy(mask), thr, mask_ms=detection_ms)
                assessment = assess(page.image, result, evidence, thr)
            result.processing['candidate_name'] = name
            results[name] = result
            records.append(dict(name=name, status='completed', method=mask.method,
                                repair_mode=mask.repair_mode, assessment=assessment))
        except Exception as exc:
            records.append(dict(name=name, status='error', error=f'{type(exc).__name__}: {exc}'))
        if len(records) == 1 and not identical:
            other = candidates[1][0]
            fallback_reason = retry_reason(records[0], source_scores[name], source_scores[other])
            if fallback_reason is None:
                records.append(dict(name=other, status='not_run',
                                    reason='未发现需要第二次修复且另一方法更有依据的条件'))
                break
    if not results:
        raise RuntimeError('所有处理候选均失败：'+'；'.join(r.get('error', '') for r in records))
    retained_source = False
    try:
        chosen, reason = choose(records, initial)
        if repair_count == 1:
            reason = initial_reason
    except RuntimeError:
        chosen = preferred if preferred in results else next(iter(results))
        reason = '所有可用候选都越过修复许可，保留原图并标为未达标'
        retained_source = True
    selected = results[chosen]
    assessment = next(r['assessment'] for r in records if r['name'] == chosen)
    unresolved = (retained_source or any(r['status'] == 'error' or r.get('assessment', {}).get('outside_permission_pixels', 0)
                      for r in records)
                  or assessment['missing_readable_items'] not in (0, None)
                  or assessment['observed_pattern_persistence'] is not None
                  and assessment['observed_pattern_persistence'] > PERSISTENCE_WARNING
                  or assessment['source_line_loss'] is not None and assessment['source_line_loss'] > .10)
    report = deepcopy(selected.verdict)
    report.elapsed_ms = mask_ms+int((perf_counter()-start)*1000)
    report.stage_ms = dict(auto_detection_ms=detection_ms,
                          auto_repair_and_comparison_ms=max(0, report.elapsed_ms-detection_ms))
    report.conditions['time_budget'] = report.elapsed_ms <= thr.time_budget_ms
    report.conditions['automatic_selection_verified'] = False
    if not report.conditions['time_budget'] and FailureCode.OUT_OF_TIME_BUDGET not in report.failures:
        report.failures.append(FailureCode.OUT_OF_TIME_BUDGET)
    if unresolved:
        report.failures.append(FailureCode.AUTO_SELECTION_UNRESOLVED)
    metadata = dict(version='source_preselection_v2', experimental=True, chosen=chosen, reason=reason,
                    status='unmet' if unresolved else 'needs_review', verified_success=False,
                    reference_used=False, residual_quality=None, content_restoration_quality=None,
                    records=records, initial=initial, source_scores=source_scores,
                    repair_count=repair_count, fallback_reason=fallback_reason)
    if identical:
        metadata['reason'] = '两种方法完全相同，复用一次处理结果；质量仍待复核'
    metadata['source_retained'] = retained_source
    support = np.zeros(page.image.shape[:2], bool)
    for y, x, *_ in evidence.patterns:
        support[y, x] = True
    metadata['source_observations'] = dict(
        readable=[dict(box=box.tolist(), text=text) for box, text, _ in evidence.readable],
        patterns=[dict(channel=p[2], operation=int(p[3]), kernel=p[4], pixels=len(p[0]))
                  for p in evidence.patterns],
        note='共同评分观测，不是真值，不提供修复权限')
    return replace(selected, verdict=report,
                   output=replace(selected.output, image=page.image.copy()) if retained_source else selected.output,
                   processing=dict(selected.processing, automatic_selection=metadata),
                   comparison_results=list(results.values()),
                   comparison_evidence={'source_pattern_observations': support})
