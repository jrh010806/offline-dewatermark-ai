"""Inventory user inputs without treating annotations or clean references as input."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tools.regression_guard import canonical, digest

MEDIA = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pdf'}


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def build_inventory(project: Path, inputs: Path) -> dict:
    cases, covered = [], set()
    source = json.loads((inputs / '输入来源与校验.json').read_text())
    catalogs = {}
    for name in ('watermark_training_v1', 'watermark_training_increment_v1_1'):
        root = project / 'data' / name
        for row in read_rows(root / 'manifest.jsonl'):
            catalogs[row['id']] = (root, row)

    def add(key, path, *, batch, group, style, negative=False, pages=1, refs=None,
            dpi=200, source_group=None, split_group=None, base_id=None, split=None, label=None):
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f'输入缺失：{path}')
        refs = refs or [{} for _ in range(pages)]
        if len(refs) != pages:
            raise ValueError(f'参考页数不一致：{key}')
        row = dict(id=key, input=str(path), input_sha256=digest(path), batch=batch,
                   group=group, style=style, label=label or style, negative=negative,
                   page_count=pages, dpi=dpi, source_group=source_group,
                   split_group=split_group, base_id=base_id, original_split=split,
                   references=refs, role='historical_regression_not_independent')
        row['reference_identity'] = canonical(refs)
        cases.append(row)
        covered.add(path)

    def reference(clean, alpha=None):
        result = {'clean': str(Path(clean).resolve()), 'clean_sha256': digest(clean)}
        if alpha:
            result.update(alpha=str(Path(alpha).resolve()), alpha_sha256=digest(alpha))
        return result

    for subset in source['sets']:
        for item in subset['files']:
            path = inputs / item['input']
            if digest(path) != item['sha256']:
                raise ValueError(f'原始输入校验失败：{path}')
            root, row = catalogs[item['id']]
            paths = row['paths']
            add(item['input'], path, batch=str(Path(item['input']).parent), group=subset['set'],
                style=item['style'], negative=item['negative'],
                refs=[reference(root / paths['clean'], root / paths['alpha'])],
                source_group=row.get('source_group'), split_group=row.get('split_group'),
                base_id=row.get('base_id'), split=row.get('split'), label=row.get('label'))

    # Explicit user-provided historical directory; do not restore retired data/real_samples.
    legacy = inputs / 'real_samples'
    manifest = json.loads((legacy / 'manifest.json').read_text())
    for row in manifest['samples']:
        add('历史图片/' + row['id'], legacy / row['jpg'], batch='历史图片/' + row['style'],
            group='截图中的历史样本', style=row['style'], source_group=row['base'],
            refs=[reference(legacy / row['clean'], legacy / 'mask' / (row['id'] + '_veil.npz'))])
        covered.add((legacy / row['clean']).resolve())
        covered.add((legacy / row['mask']).resolve())
    for style, path in manifest['style_pdfs'].items():
        matches = [r for r in manifest['samples'] if r['style'] == style]
        add('历史PDF/' + style, legacy / path, batch='历史PDF/' + style,
            group='截图中的历史样本', style=style, pages=len(matches),
            refs=[reference(legacy / r['clean'], legacy / 'mask' / (r['id'] + '_veil.npz')) for r in matches])

    # Keep existing fixed PDF/negative/style coverage in addition to the screenshot inputs.
    root = project / 'data/watermark_regression_min_v1'
    from tools.run_min_regression import validate
    fixed, _ = validate(root)
    for row in fixed:
        refs = []
        for page_number, p in enumerate(row['pages'] if row['kind'] == 'pdf' else [row], 1):
            paths = p['paths']
            ref = (reference(root / paths['clean'], root / paths['alpha'])
                   if row['score_mode'] == 'paired' else {})
            if ref and row['kind'] == 'pdf':
                clean_pdf = root / row['paths']['clean']
                ref.update(clean_pdf=str(clean_pdf.resolve()), clean_pdf_sha256=digest(clean_pdf),
                           page_number=page_number, dpi=96)
            refs.append(ref)
        add('固定回归/' + row['case_id'], root / row['paths']['input'],
            batch='固定回归/' + row['case_id'], group='固定回归', style=row['subtype'],
            pages=row['page_count'], negative=row['negative'], refs=refs, dpi=96,
            source_group=row['source_group'], split=row['original_split'], label=row['label'])

    extras = [str(p.relative_to(inputs)) for p in inputs.rglob('*')
              if p.suffix.lower() in MEDIA and p.resolve() not in covered]
    if extras:
        raise ValueError(f'存在未登记的输入或参考，必须先分类：{extras}')
    if len({r['id'] for r in cases}) != len(cases):
        raise ValueError('输入场景 ID 重复')
    return dict(schema_version=1, input_root=str(inputs.resolve()), cases=cases,
                groups=dict(Counter(r['group'] for r in cases)),
                total_inputs=len(cases), total_pages=sum(r['page_count'] for r in cases),
                unique_input_hashes=len({r['input_sha256'] for r in cases}),
                independent_validation=False,
                note='重复文件保留其不同批次场景；参考只供处理后评分。历史与已查看样本均不能证明独立泛化。')
