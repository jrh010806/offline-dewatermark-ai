"""Read-only, post-inference effect triage of a sealed observation baseline.

These presentation categories are not regression gates or quality acceptance.
No detector, repairer, OCR engine or model is run by this report generator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageOps
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regression_guard import canonical, digest
from tools.regression_retirements import load_profile, project_baseline

RULES = {
    'version': '效果初筛V1_20261029',
    'purpose': '展示分档，不修改算法、不修改防退化门槛，不作为质量验收或成功率',
    'good': {'recall_min': .95, 'precision_min': .95, 'watermark_mae_max': 5,
             'watermark_error_reduction_min': .8, 'background_damage_mae_max': .05,
             'hole_selected_pixels_max': 0, 'outside_selection_changed_pixels_max': 0},
    'poor': {'background_damage_mae_gt': 1, 'background_damage_p99_gt': 12,
             'error_reduction_lt': .3, 'watermark_mae_ge': 5},
    'unknown': '没有严格配对、原图差异太小或条件不足时归入一般／待复核，未知质量保持null',
    'negative': '无水印且选区和改动均为0，标为无水印安全保留；不计作去水印成功',
    'deduplication': '同模式、同输入文件哈希、同页码合并；不同批次结果不一致时展示最差档，全部原始记录保留',
}
BUCKETS = {'good': '效果较好（初筛）', 'review': '一般／待复核', 'poor': '问题较多（初筛）'}
RANK = {'good': 0, 'review': 1, 'poor': 2}


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def page_rgb(path, page, dpi):
    if Path(path).suffix.lower() != '.pdf':
        return rgb(path)
    with pymupdf.open(path) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def measured_comparison(record, case):
    ref = case['references'][record['page'] - 1]
    if not ref.get('clean') or not ref.get('alpha') or record['status'] != 'completed':
        return dict(source_watermark_mae=None, watermark_error_reduction=None,
                    background_damage_mae=None, background_damage_p99=None, exact_reference_match=None)
    output = rgb(record['files']['处理结果.png'])
    source = page_rgb(case['input'], record['page'], case['dpi'])
    clean = (page_rgb(ref['clean_pdf'], ref['page_number'], ref['dpi'])
             if ref.get('clean_pdf') else rgb(ref['clean']))
    if Path(ref['alpha']).suffix.lower() == '.npz':
        with np.load(ref['alpha'], allow_pickle=False) as data:
            alpha = data['alpha'].copy()
    else:
        with Image.open(ref['alpha']) as image:
            alpha = np.asarray(image.convert('L').resize((output.shape[1], output.shape[0]), Image.Resampling.LANCZOS))
    if source.shape != output.shape or clean.shape != output.shape or alpha.shape != output.shape[:2]:
        raise ValueError('参考或输入的几何不一致，不可进行像素分档')
    truth = alpha > 0
    before = np.abs(source.astype(np.int16) - clean.astype(np.int16)).mean(2)
    after = np.abs(output.astype(np.int16) - clean.astype(np.int16)).mean(2)
    source_mae = float(before[truth].mean()) if truth.any() else None
    result_mae = float(after[truth].mean()) if truth.any() else None
    if record.get('watermark_mae') is not None and abs(result_mae - record['watermark_mae']) > .0001:
        raise ValueError('事后复算与封存的水印区域误差不一致')
    damage = np.maximum(after[~truth] - before[~truth], 0)
    return dict(source_watermark_mae=source_mae,
                watermark_error_reduction=(1 - result_mae / source_mae) if source_mae and source_mae > 1 else None,
                background_damage_mae=float(damage.mean()) if damage.size else None,
                background_damage_p99=float(np.percentile(damage, 99)) if damage.size else None,
                exact_reference_match=bool(np.array_equal(output, clean)))


def classify(row):
    if row.get('status') != 'completed':
        return 'poor', ['运行或评分异常，未形成可确认的完整结果']
    if row['negative']:
        if row['selected_pixels'] == 0 and row['changed_pixels'] == 0:
            return 'good', ['无水印对照：没有新增选区，输出保持原样；不代表去水印成功']
        return 'poor', [f'无水印对照被误选或改动：选中{row["selected_pixels"]:,}像素，改动{row["changed_pixels"]:,}像素']
    if row.get('exact_reference_match') is True:
        return 'good', ['本页实际输出与对应干净参考逐像素一致；只适用于这张／页']
    if row.get('watermark_mae') is None or row.get('exact_reference_match') is None:
        return 'review', ['没有严格配对参考，不能可靠评定去除效果；请查看实际原图与结果']
    reasons = []
    if row.get('background_damage_mae', 0) > RULES['poor']['background_damage_mae_gt']:
        reasons.append('非水印区域对参考的误差明显增加，存在内容损伤风险')
    if row.get('background_damage_p99', 0) > RULES['poor']['background_damage_p99_gt']:
        reasons.append('非水印区域局部误差增加较大，需重点核对正文和纹理')
    reduction = row.get('watermark_error_reduction')
    if (row['changed_pixels'] == 0 or row['selected_pixels'] == 0) and (row.get('source_watermark_mae') or 0) > 1:
        reasons.append('带水印输入未得到有效处理，或未形成水印选区')
    elif reduction is not None and reduction <= 0:
        reasons.append('水印区域对干净参考的像素误差没有减少，或反而增加')
    elif reduction is not None and reduction < .3 and row['watermark_mae'] >= 5:
        reasons.append('水印区域对参考的误差改善不足，结果仍有较大差异')
    if reasons:
        return 'poor', reasons
    good = RULES['good']
    if (row.get('recall') is not None and row['recall'] >= good['recall_min'] and
            row.get('precision') is not None and row['precision'] >= good['precision_min'] and
            row['watermark_mae'] <= good['watermark_mae_max'] and
            reduction is not None and reduction >= good['watermark_error_reduction_min'] and
            (row.get('background_damage_mae') or 0) <= good['background_damage_mae_max'] and
            (row.get('hole_selected_pixels') or 0) == 0 and row['outside_selection_changed_pixels'] == 0):
        return 'good', ['笔画覆盖和选区精确率均≥95%，水印区域对参考误差减少≥80%，背景新增平均误差≤0.05',
                        '这是自动指标初筛，残影、正文正确性和纹理仍需目视核对']
    if (row.get('recall') or 0) < .95:
        reasons.append('有水印像素未被选中，需查看淡笔画、边缘或交叠处')
    if (row.get('precision') or 0) < .95:
        reasons.append('选区包含背景像素，需核对是否造成误伤')
    if row.get('hole_selected_pixels'):
        reasons.append('真实字洞内存在选区，需核对处理结果')
    if row['watermark_mae'] > 5:
        reasons.append('水印区域与干净参考仍有差异，可能包含残影或重建差异')
    if reduction is None:
        reasons.append('原图对参考的差异过小，无法可靠计算改善比例')
    return 'review', reasons or ['有一定改善，但未同时满足较好档的初筛条件']


def main():
    ap = argparse.ArgumentParser(description='把完整观察基线按实际结果初筛为三个效果标签；不运行算法')
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    baseline = (args.baseline or Path(json.loads((ROOT / '.regression-control/baseline.json').read_text())['path'])).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '预览图').mkdir()
    run = json.loads((baseline / '运行记录.json').read_text())
    inventory = json.loads((baseline / '样本清单.json').read_text())
    if not run['complete'] or not run['integrity_verified'] or len(run['records']) != len(run['expected_keys']):
        raise ValueError('原始运行未完整记录，不能冒充完整效果总览')
    source_hash = digest(baseline / '运行记录.json')
    run, _ = project_baseline(run, json.loads((baseline / '原始清单.json').read_text()),
                              load_profile(), digest(baseline / '封存校验.json'))
    cases = {r['id']: r for r in inventory['cases']}
    def url(path):
        return quote(os.path.relpath(path, out).replace(os.sep, '/'), safe='/')
    cache, preview_cache, rows = {}, {}, []
    for number, record in enumerate(run['records'], 1):
        row = {k: record.get(k) for k in ('key', 'id', 'page', 'mode', 'status', 'input_sha256',
             'style', 'group', 'negative', 'recall', 'precision', 'strong_recall', 'missed_pixels',
             'false_positive_pixels', 'hole_selected_pixels', 'selected_pixels', 'selected_fraction',
             'changed_pixels', 'outside_selection_changed_pixels', 'elapsed_ms', 'watermark_mae',
             'background_mae', 'method', 'repair_mode', 'error')}
        row.update(residual_quality=None, recognition_quality=None, manual_quality_reviewed=False)
        case = cases[row['id']]
        key = canonical([record.get('output_sha256'), row['input_sha256'], row['page'], case['reference_identity']])
        if key not in cache:
            cache[key] = measured_comparison(record, case)
        row.update(cache[key])
        row['bucket'], row['reasons'] = classify(row)
        row['label'] = case.get('label') or row['style']
        row['original'] = url(case['input'])
        files = record.get('files', {})
        row['source_preview'] = url(files['原图预览.jpg']) if files.get('原图预览.jpg') else None
        row['selection_preview'] = url(files['选区预览.jpg']) if files.get('选区预览.jpg') else None
        row['diagnostic'] = url(files['漏选多选核对.jpg']) if files.get('漏选多选核对.jpg') else None
        row['output'] = url(files['处理结果.png']) if files.get('处理结果.png') else None
        if row['output']:
            output_hash = record['output_sha256']
            if output_hash not in preview_cache:
                target = out / '预览图' / (output_hash[:24] + '.jpg')
                with Image.open(files['处理结果.png']) as image:
                    image.thumbnail((900, 900))
                    image.convert('RGB').save(target, quality=88)
                preview_cache[output_hash] = url(target)
            row['output_preview'] = preview_cache[output_hash]
        rows.append(row)
        if number % 50 == 0:
            print(f'只读整理 {number}/{len(run["records"])}', flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['mode'], row['input_sha256'], row['page'])].append(row)
    unique_rows = []
    for occurrences in grouped.values():
        representative = max(occurrences, key=lambda r: (RANK[r['bucket']], r.get('watermark_mae') or 0))
        unique_rows.append(dict(representative, duplicate_keys=[r['key'] for r in occurrences],
                                duplicate_groups=[r['group'] for r in occurrences]))
    counts = {mode: dict(Counter(r['bucket'] for r in unique_rows if r['mode'] == mode)) for mode in run['profiles']}
    negative_counts = {mode: sum(r['negative'] and r['bucket'] == 'good' for r in unique_rows if r['mode'] == mode) for mode in run['profiles']}
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_completed_at=run['completed_at'], source_record_sha256=source_hash,
        baseline=str(baseline), code_commit=run.get('git_commit'), record_count=len(rows),
        source_record_count=len(json.loads((baseline / '运行记录.json').read_text())['records']),
        unique_page_mode_count=len(unique_rows), by_mode=counts, safe_negative_by_mode=negative_counts,
        classification_rules=RULES, manual_quality_reviewed=False, independent_validation=False,
        residual_quality=None, recognition_quality=None,
        sample_scope_revision=run.get('sample_scope_revision'),
        note='分档是事后自动指标初筛，不是视觉验收、成功率或水印残留率。不同格式不合并；重复批次保留完整原始记录。')
    (out / '效果分档规则_20261029.json').write_text(json.dumps(RULES, ensure_ascii=False, indent=2))
    (out / '逐图效果初筛_20261029.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    (out / '效果总览来源与统计_20261029.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if run.get('sample_scope_revision'):
        (out / '样本移除说明_20261029.json').write_text(json.dumps(load_profile(), ensure_ascii=False, indent=2))
    payload = dict(rows=rows, counts=counts, negative_counts=negative_counts,
                   sources=dict(original_report=url(baseline / '项目防退化控制报告.html'),
                                original_records=url(baseline / '运行记录.json'),
                                inventory=url(baseline / '原始清单.json'), rules='效果分档规则_20261029.json'),
                   manifest=manifest)
    template = (ROOT / 'tools/templates/effect_overview.html').read_text()
    page = template.replace('__REVIEWED_DATA__', json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\u003c'))
    dest = out / '去水印效果分档总览_20261029.html'
    dest.write_text(page)
    if digest(baseline / '运行记录.json') != source_hash:
        raise ValueError('源记录发生变化，不能将本次整理当作单一版本结果')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(dest)


if __name__ == '__main__':
    main()
