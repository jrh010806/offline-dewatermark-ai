"""Synchronize current task training status without rewriting historic scores."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/watermark_training_increment_v1_1'


def sync():
    qa_path=D/'qa/training_tasks_validation.json'
    if not qa_path.exists():return
    qa=json.loads(qa_path.read_text())
    assert qa['status']=='passed'
    assert hashlib.sha256((D/'training_tasks.jsonl').read_bytes()).hexdigest()==qa['task_manifest_sha256']
    text_note='''<div id="current-real-training-status" style="background:#eaf5ef;border:1px solid #b5d5c2;border-radius:10px;padding:18px;margin:16px 0"><strong>当前训练状态：</strong>人工增量统一清单已有 <b>129 个不重复训练样本</b>：原 121 条严格配对、3 组低权重近似修复配对、5 张检测专用图（4 张实扫＋1 张红笔批注）。修复视图 124 条，检测视图 126 条，两个任务重用原 121 条，不能相加为独立样本数。8 个真实样本均已完成相应用途的标注、视觉核对和加载检查。<br>扫描图仅提供检测框，不冒充精确分割或修复真值；红笔单图没有干净目标。原 121 条严格清单仍保留兼容读取。<br><a href="真实样本训练标注.html">查看全部 8 个真实样本、标注和参考图 ↗</a> · <a href="../data/watermark_training_increment_v1_1/training_tasks.jsonl">129 条统一训练清单</a> · <a href="../data/watermark_training_increment_v1_1/qa/training_tasks_validation.json">训练读取校验</a></div>'''
    replacements=[
        ('待配准及掩码复核','已完成配准与检测框核对，检测专用'),
        ('配准与水印区域标注复核待做','配准与检测框标注已完成，已纳入检测训练'),
        ('尚未完成透视配准与水印掩码，不进入默认像素监督训练','已完成配准和水印检测框标注；检测专用，不作为精确像素修复配对'),
        ('辅助集，不进默认像素监督训练','2 组已纳入低权重近似修复训练；单图已纳入红笔检测训练'),
        ('已有配准参考，单列辅助使用','已纳入近似修复训练，权重 0.25；不用于精确像素评测'),
        ('不由默认监督训练加载器读取；两张抖音图及登记已删除','5 张均已进入检测训练；两张抖音图及登记已删除'),
        ('接下来完成配准与水印区域标注复核','已完成检测标注及加载检查'),
        ('3 组近似配对与 5 条参考','3 组近似修复配对与 5 张检测训练样本'),
        ('默认读取器只读取已入库的 121 条','默认读取器读取统一任务视图 129 条；严格配对视图保留原 121 条'),
        ('默认读取器不混入精确像素监督清单','统一读取器以 0.25 权重读取这 3 组近似配对，不将其冒称为精确真值'),
        ('weak_pairs.jsonl 和 references.jsonl 不会被默认训练读取器纳入','weak_pairs.jsonl 和 references.jsonl 保留来源快照；其 8 个真实样本已按 training_tasks.jsonl 纳入相应训练任务'),
        ('透视配准和掩码 QA 完成前不并入默认训练清单','已完成配准与检测框视觉核对，并纳入检测训练；没有同拍摄条件的修复目标'),
        ('电子源页不是同拍摄条件干净真值；纸张透视和弯曲尚未标注处理','电子源页不是同拍摄条件干净真值；现已关联检测框标注，纸张弯曲仍是精度限制'),
    ]
    for p in (ROOT/'docs').glob('*.html'):
        if p.name in {'首轮基线评测.html','真实样本训练标注.html','最小回归测试集.html'}:continue
        s=p.read_text();s=re.sub(r'<div id="current-real-training-status".*?</div>','',s,flags=re.S)
        for a,b in replacements:s=s.replace(a,b)
        s=re.sub(r'(<main\b[^>]*>)',lambda m:m[1]+text_note,s,count=1)
        p.write_text(s)
    card=D/'DATASET_CARD.html'
    if card.exists():
        s=card.read_text();s=re.sub(r'<div id="current-real-training-status".*?</div>','',s,flags=re.S)
        for a,b in replacements:s=s.replace(a,b)
        portable_note=text_note.replace('href="真实样本训练标注.html"','href="../../docs/真实样本训练标注.html"').replace('href="../data/watermark_training_increment_v1_1/','href="')
        s=re.sub(r'(<main\b[^>]*>)',lambda m:m[1]+portable_note,s,count=1);card.write_text(s)
    print('Current task training status synchronized in HTML documents')


if __name__=='__main__':sync()
