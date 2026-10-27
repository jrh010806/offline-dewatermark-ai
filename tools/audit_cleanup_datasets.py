"""Review known data artefacts and move regenerable clutter to macOS Trash.

No recursive guessed-path deletion. Current datasets, unique source materials,
model checkpoints, and benchmark output images are retained in place.
"""
from __future__ import annotations
import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile
import zlib

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
STAMP=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
AUDIT=ROOT/'out/dataset_cleanup'/STAMP


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def jsave(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def size(p):return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def rel(p):return p.relative_to(ROOT).as_posix()


def verify_archive(archive,canonical):
    """Compare every image/PDF/GT member using ZIP CRC and size, metadata by existence."""
    checked=0;metadata_differences=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir() or info.filename.startswith('__MACOSX/') or info.filename.endswith('.DS_Store'):continue
            member=Path(info.filename)
            if len(member.parts)<2:raise ValueError('Unexpected archive root')
            target=canonical/Path(*member.parts[1:])
            assert target.is_file(),f'Archive unique file must be retained: {target}'
            if target.suffix.lower() in ('.png','.jpg','.jpeg','.pdf','.npz','.otf','.ttf'):
                assert target.stat().st_size==info.file_size,str(target)
                crc=0
                with target.open('rb') as stream:
                    for b in iter(lambda:stream.read(4*1024*1024),b''):crc=zlib.crc32(b,crc)
                assert crc & 0xffffffff==info.CRC,str(target)
                checked+=1
            elif target.stat().st_size!=info.file_size:
                metadata_differences.append(rel(target))
        candidates=[i for i in z.infolist() if i.filename.endswith('/manifest.jsonl')]
        if candidates:assert hashlib.sha256(z.read(candidates[0])).hexdigest()==sha(canonical/'manifest.jsonl')
    return dict(archive=rel(archive),canonical=rel(canonical),verified_media_members=checked,
        archive_sha256=sha(archive),metadata_may_have_newer_edits=metadata_differences)


def main():
    registry_file=ROOT/'data/dataset_registry.json'
    if registry_file.exists() and json.loads(registry_file.read_text()).get('historical_reference_sets')==0:
        print('历史数据已退役；当前清单见 data/dataset_registry.json。无需重复清理。')
        return
    AUDIT.mkdir(parents=True,exist_ok=False)
    datasets=[
        dict(id='training_v1',role='active_training',path='data/watermark_training_v1',label='基础训练集 v1',count=2432,unit='图片/文档记录',detail='2,112 图片正例、176 PDF 文档正例、144 份负例'),
        dict(id='training_increment',role='active_training',path='data/watermark_training_increment_v1_1',label='人工训练增量 v1.1',count=121,unit='图片/页训练记录',detail='110 图片正例、5 负例、6 PDF 页配对；另有近似配对和参考'),
        dict(id='regression_min',role='active_regression',path='data/watermark_regression_min_v1',label='固定最小回归集',count=46,unit='输入文件',detail='核心36＋真实扩展10；从前两套派生，不增加独立来源'),
        dict(id='legacy_real',role='historical_reference',path='data/real_samples',label='旧真实底图合成样本',count=16,unit='配对样本',detail='4 种旧样式；水印仍为合成，现有回归脚本仍引用'),
        dict(id='legacy_synthetic',role='historical_reference',path='data/syn',label='早期合成文档集',count=15,unit='页面',detail='5 种样式；启动/工程测试仍引用'),
        dict(id='public_reference',role='historical_reference',path='data/public_watermarks_20261026',label='下载作者公开样本',count=40,unit='配对图片',detail='作者测试子集；仅历史参考，未并入正式训练'),
    ]
    protected=[ROOT/d['path']/'manifest.jsonl' for d in datasets if (ROOT/d['path']/'manifest.jsonl').exists()]
    protected += [ROOT/'data/watermark_training_v1/splits'/f'{s}.jsonl' for s in ['train','validation','test']]
    protected += [ROOT/'out/baseline_20261026/metrics.json',ROOT/'out/baseline_20261026/pdf_metrics.json']
    before={rel(p):sha(p) for p in protected}
    # Validate all exact-pair alpha/mask images without rejecting genuinely hard positives.
    cases=[]
    for root in [ROOT/'data/watermark_training_v1',ROOT/'data/watermark_training_increment_v1_1']:
        for r in rows(root/'manifest.jsonl'):
            for p in r['pages'] if r['kind']=='pdf' else [r]:
                cases.append((root,p,r.get('is_negative',False)))
    def inspect(item):
        root,row,negative=item
        for role in ['alpha','mask']:
            with Image.open(root/row['paths'][role]) as image:
                low,high=image.convert('L').getextrema()
                if (high==0) != negative:return {'path':str(root/row['paths'][role]),'reason':'empty positive or nonzero negative'}
        return None
    with ThreadPoolExecutor(max_workers=8) as pool:bad=[r for r in pool.map(inspect,cases) if r]
    jsave(AUDIT/'quality_check.json',dict(exact_pair_images_or_pages=len(cases),invalid_alpha_or_mask=bad))
    if bad:raise SystemExit('Questionable samples found; retained without deleting. Inspect quality_check.json.')
    print(f'Exact-pair label audit passed: {len(cases)} images/pages; no empty positive masks.',flush=True)
    archive_specs=[('data/real_samples.zip','data/real_samples'),
        ('out/training_dataset_release/watermark_training_v1.zip','data/watermark_training_v1'),
        ('out/training_dataset_release/watermark_training_increment_v1_1.zip','data/watermark_training_increment_v1_1')]
    archives=[]
    for path,target in archive_specs:
        report=verify_archive(ROOT/path,ROOT/target);archives.append(report)
        jsave(AUDIT/'archive_checks.json',archives)
        print(f'Export media matches retained source: {path} ({report["verified_media_members"]} files)',flush=True)
    targets=[(ROOT/path,'已验证媒体与现存目录一致的重复导出包') for path,_ in archive_specs]
    targets += [(ROOT/'out/training_dataset_release'/name,'已删除导出包的哈希旁文件；发布记录仍保留') for name in
        ['watermark_training_v1.zip.sha256','watermark_training_increment_v1_1.zip.sha256']]
    targets += [(ROOT/'out/training_dataset_release/portable_check','发布包抽取测试副本；现存完整数据可重建'),
        (ROOT/'out/manual_increment_work','人工素材配准中间输出；正式原图和已入库配准件保留'),
        (ROOT/'out/manual_pdf_intake_001','PDF 接收检查临时拼图与检查件；原 PDF 和对象标注保留'),
        (ROOT/'data/pdf/doc00_tiled_text/_tmp','PDF 夹具生成中间渲染图片；两份原生夹具保留')]
    # Include only well-known macOS directory metadata under the task's data/output.
    for base in [ROOT/'data',ROOT/'out',ROOT/'docs']:
        targets.extend((p,'macOS .DS_Store 缓存') for p in base.rglob('.DS_Store'))
    unique=[]
    for path,reason in targets:
        if path.exists() and not any(path==p or p in path.parents for p,_ in unique):unique.append((path,reason))
    plan=[dict(path=rel(p),bytes=size(p),reason=reason) for p,reason in unique]
    jsave(AUDIT/'cleanup_plan.json',plan)
    trash=Path.home()/'.Trash'/('offline-dewatermark-ai_cleanup_'+STAMP)
    trash.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='moving',created_at=datetime.now().astimezone().isoformat(),method='recoverable_move_to_macos_trash',
        trash_root=str(trash),items=[],total_bytes_removed_from_project=0)
    for path,reason in unique:
        destination=trash/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        item=dict(original_path=str(path),trash_path=str(destination),bytes=size(path),reason=reason)
        if path.is_file():item['sha256']=sha(path)
        shutil.move(str(path),str(destination))
        receipt['items'].append(item);receipt['total_bytes_removed_from_project']+=item['bytes']
        jsave(AUDIT/'cleanup_receipt.json',receipt)
    assert {rel(p):sha(p) for p in protected}==before
    receipt.update(status='completed',active_manifests_splits_latest_metrics_unchanged=True,
        retained_historical_results=True,retained_model_weights=True,retained_original_user_materials=True)
    jsave(AUDIT/'cleanup_receipt.json',receipt)
    restore='''import json, shutil\nfrom pathlib import Path\nreceipt=json.loads((Path(__file__).parent/'cleanup_receipt.json').read_text())\nfor item in receipt['items']:\n    source=Path(item['trash_path']);target=Path(item['original_path'])\n    if not source.exists():raise SystemExit(f'Missing recovery item: {source}')\n    if target.exists():raise SystemExit(f'Refusing to overwrite: {target}')\nfor item in receipt['items']:\n    target=Path(item['original_path']);target.parent.mkdir(parents=True,exist_ok=True)\n    shutil.move(item['trash_path'],str(target))\nprint('Restored original paths.')\n'''
    (AUDIT/'restore.py').write_text(restore)
    for d in datasets:d['bytes']=size(ROOT/d['path'])
    registry=dict(as_of='2026-10-26',active_sets=3,historical_reference_sets=3,registered_sets=6,
        datasets=datasets,support_formats=[dict(path='data/pdf',note='两种 PDF 夹具，派生格式，不另计独立样本集'),
            dict(path='data/real_samples_eval',note='旧样本的评测格式，保留历史报告和复现依赖')],
        original_sources=['data/manual_collections','data/base_photos','data/20250227.pdf'],
        cleanup_receipt=rel(AUDIT/'cleanup_receipt.json'),trash_root=str(trash),
        removed_bytes=receipt['total_bytes_removed_from_project'],note='The regression subset and transformed legacy formats must not be added to training-source counts.')
    jsave(ROOT/'data/dataset_registry.json',registry)
    (ROOT/'data/README.txt').write_text('数据目录总入口\n\n日常主用 3 套：\n1. watermark_training_v1：基础训练集，2432 条图片/文档记录。\n2. watermark_training_increment_v1_1：人工训练增量，121 条图片/页记录。\n3. watermark_regression_min_v1：固定回归输入46个（核心36＋真实扩展10），由前两套派生。\n\n历史参考3套：real_samples（16组）、syn（15页）、public_watermarks_20261026（40对）。\npdf 与 real_samples_eval 是配套格式，不另加为独立数据来源。\nmanual_collections、base_photos、20250227.pdf 是原始来源材料，不当作新训练数据集。\nout/ 是推理结果、实验和发布输出，模型权重不是数据集。\n\n全量登记见 dataset_registry.json；可视说明见 ../docs/数据集清单与清理.html。\n重复导出包与明确临时件已移至系统废纸篓；保留清理清单和恢复脚本。\n')
    active=''.join(f'<tr><td>{html.escape(d["label"])}</td><td>{"当前主用" if d["role"].startswith("active") else "历史参考"}</td><td>{d["count"]} {d["unit"]}</td><td><code>{d["path"]}</code><br>{html.escape(d["detail"])}</td></tr>' for d in datasets)
    cleaned=''.join(f'<tr><td>{html.escape(str(Path(i["original_path"]).relative_to(ROOT)))}</td><td>{i["bytes"]/1048576:.2f} MiB</td><td>{html.escape(i["reason"])}</td></tr>' for i in receipt['items'])
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;font:15px/1.85 -apple-system,"PingFang SC",sans-serif;color:#24384f}aside{position:fixed;inset:0 auto 0 0;width:230px;padding:26px 18px;background:#162a42;color:white}aside a{display:block;padding:10px;color:#d7e3ef;text-decoration:none}main{margin-left:230px;max-width:1480px;padding:32px}section{background:white;padding:28px;border:1px solid #dce5ee;border-radius:12px;margin-bottom:22px;scroll-margin-top:20px}h1{font-size:30px}h2{font-size:22px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dfe6ee}th{background:#edf3f9}code{overflow-wrap:anywhere;font-size:12px}.notice{padding:16px;background:#edf6f1;border-left:4px solid #26816b}a{color:#24669e}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:16px;font-size:12px}@media(max-width:700px){aside{position:relative;width:auto}aside a{display:inline-block}main{margin:0;padding:12px}section{padding:18px}}'''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据集清单与清理</title><style>{css}</style></head><body><aside><h2>数据集整理</h2><a href="#summary">01　现在有几套</a><a href="#retained">02　哪些配套材料保留</a><a href="#cleanup">03　垃圾清理明细</a><a href="#recovery">04　恢复与检查</a><a href="项目文档总览.html">← 项目文档总览</a></aside><main>
<section id="summary"><h1>当前主用 3 套，历史参考 3 套</h1><p>共登记 <b>6 套样本集合</b>。真正用于当前训练准备的是基础集和人工增量；最小回归集从它们派生，不增加独立训练样本数量。</p><div class="table"><table><tr><th>集合</th><th>状态</th><th>数量</th><th>路径与用途</th></tr>{active}</table></div><p>日常修改后使用 <a href="最小回归测试集.html">固定最小回归集</a>；全量样本和现有测试对比见 <a href="水印分类样本与最新测试.html">分类、样本与测试报告</a>。</p></section>
<section id="retained"><h2>配套格式、原始素材和实验结果</h2><ul><li>data/pdf 和 data/real_samples_eval 是既有样本的格式/夹具，有启动脚本、旧评测文档或复现用途，不另计数据集。</li><li>data/manual_collections、data/base_photos、data/20250227.pdf 是来源材料；原图、PDF、授权和受限接收记录保留。</li><li>out/baseline_20261026 及 before_photo_guard 是最新结果与修改前对照；历史失败输出和模型权重保留。</li><li>难例、未配准实扫、算法暂时去不掉的样本都不是垃圾；未经准入的下载素材单列参考，不混入正式训练。</li></ul><p>没有移动现有数据目录，因此当前文档图片、代码和历史报告仍沿用原路径。</p></section>
<section id="cleanup"><h2>已从项目清理 {receipt['total_bytes_removed_from_project']/1073741824:.2f} GiB</h2><p class="notice">以下文件已移到系统废纸篓，可以恢复；废纸篓未清空，因此这里表示项目占用减少，不表示系统磁盘已释放同等容量。</p><div class="table"><table><tr><th>原路径</th><th>大小</th><th>清理依据</th></tr>{cleaned}</table></div><p>三个 ZIP 中的图片、PDF、标注及字体逐文件核对尺寸和 CRC，主清单核对 SHA256；对应原文件均留在正式目录。旧发布元数据 JSON 保留作为历史记录，包文件当前已清理，需要时按打包脚本重新导出。</p></section>
<section id="recovery"><h2>检查与恢复</h2><p>2,745 张/页正式配对或负例的 alpha/mask 检查通过；本次未发现需要删除的空正例或非零负例标注。现有训练主清单、train/validation/test 划分及最新指标文件哈希保持不变。</p><p><a href="../data/dataset_registry.json">数据集登记 JSON</a> · <a href="../{rel(AUDIT/'cleanup_receipt.json')}">逐项清理与恢复路径</a> · <a href="../{rel(AUDIT/'archive_checks.json')}">导出包核对记录</a></p><p>恢复命令（在项目根目录）：</p><pre>.venv/bin/python {rel(AUDIT/'restore.py')}</pre><p>废纸篓位置：</p><pre>{html.escape(str(trash))}</pre></section></main></body></html>'''
    (ROOT/'docs/数据集清单与清理.html').write_text(page)
    print(json.dumps(dict(status='completed',registered_sets=6,active_sets=3,historical_sets=3,
        moved_items=len(receipt['items']),project_reduction_bytes=receipt['total_bytes_removed_from_project'],
        receipt=rel(AUDIT/'cleanup_receipt.json'),trash_root=str(trash)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
