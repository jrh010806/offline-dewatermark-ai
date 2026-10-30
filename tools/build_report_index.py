"""Organize local reports into offline searchable tabs; never touch sealed runs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape, unescape
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
INDEX = OUT / '水印验证报告总目录.html'
CATEGORIES = {
    'baseline': ('观察基线', '保存当时版本的表现，用于后续逐张比较；记录完成不代表当前代码全量通过或去水印质量合格。'),
    'control': ('项目管控', '版本冻结、防退化规则、工具自检和工程记录。工具自检只证明控制链路可运行。'),
    'diagnosis': ('失败与诊断', '已记录的失败案例、定位分析与瓶颈判断；保留未修复和待验证事项。'),
    'experiment': ('算法实验', '定位、笔画补全及修复方法的实验对照。每轮的适用条件和验收结论以原报告为准。'),
    'history': ('历史回归', '此前样本回归、旧版本结果和错误示例。历史结果不能当作当前成绩或独立验证。'),
    'material': ('方案材料', '技术方案、可编辑 Word / PPT 和交付说明；材料更新不代表新增算法验证。'),
    'all': ('全部报告', '查看所有分类中的报告入口，或搜索标题、说明和目录路径。'),
}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.links = []
        self.title = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.in_title = True
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href')

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        self.parts.append(data)
        if self.in_title:
            self.title.append(data)


def plain(value):
    parser = Text()
    parser.feed(value)
    return ' '.join(parser.parts)


def href(path):
    import os
    return quote(Path(os.path.relpath(path, OUT)).as_posix(), safe='/')


def clean_document_links(body):
    """Local document entry points end at the real extension; keep page-only anchors."""
    extensions = {'.html', '.htm', '.md', '.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.json', '.csv'}
    def replace(match):
        value = unescape(match.group(2))
        parts = urlsplit(value)
        if parts.scheme not in ('', 'file') or not parts.path:
            return match.group(0)
        if Path(unquote(parts.path)).suffix.lower() not in extensions:
            return match.group(0)
        clean = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        return 'href=' + match.group(1) + escape(clean, quote=True) + match.group(1)
    return re.sub(r'''href=(["'])(.*?)\1''', replace, body)


def available_local_links(body, base=None):
    """Keep missing files visible without publishing a broken clickable link.

    Retain the original target so a later rebuild can restore the same link
    when that exact file returns. Never edit or substitute historical assets.
    """
    base = OUT if base is None else Path(base)
    note = '<small data-missing-note="true">（本地文件缺失）</small>'

    def replace(match):
        attrs, inner = match.group(1), match.group(2)
        active = re.search(r'''(?<![\w-])href\s*=\s*(["'])(.*?)\1''', attrs)
        dormant = re.search(r'''data-missing-href\s*=\s*(["'])(.*?)\1''', attrs)
        original = active or dormant
        if original is None:
            return match.group(0)
        value = unescape(original.group(2))
        parts = urlsplit(value)
        if parts.scheme or not parts.path:
            return match.group(0)
        exists = (base / unquote(parts.path)).exists()
        if active is not None and exists:
            return match.group(0)
        if dormant is not None and not exists:
            return match.group(0) if note in inner else f'<a{attrs}>{inner}{note}</a>'
        attrs = attrs[:original.start()] + attrs[original.end():]
        attrs = re.sub(r'''\s+aria-disabled\s*=\s*(["'])true\1''', '', attrs)
        if exists:
            return f'<a{attrs} href="{escape(value, quote=True)}">{inner.replace(note, "")}</a>'
        return (f'<a{attrs} data-missing-href="{escape(value, quote=True)}" '
                f'aria-disabled="true">{inner}{note}</a>')

    return re.sub(r'<a\b([^>]*)>(.*?)</a>', replace, body, flags=re.S)


def category(value):
    if '完整基线效果总览' in value or '效果分档总览' in value:
        return 'baseline'
    if any(k in value for k in ('tender-', '方案材料｜', '答辩材料｜', 'proposal/output')):
        return 'material'
    if any(k in value for k in ('项目防退化控制', '分层防退化检查', '项目整理', '样本移除记录', 'project-control', '空间清理审计')):
        return 'control'
    if any(k in value for k in ('失败案例', 'bottleneck', 'failure-cases', '瓶颈判断', '漏检原因分析',
                                'batch_diagnosis', 'report_before.html', 'export_selection_diagnosis')):
        return 'diagnosis'
    if any(k in value for k in ('历史错误', '旧版错误', 'recall_priority', '历史示意版', '早期分析',
                                'center_refinement', 'new_samples', 'app_quick20', 'small_dataset_rerun',
                                'user_optimization', '当前默认基线', '指标数据｜', 'stroke_selection_20261028/final')):
        return 'history'
    return 'experiment'


def card(identifier, group, body):
    return dict(id=identifier, category=group, body=body)


def original_cards(source):
    # Subsequent rebuilds retain curated explanations and any newly added entries.
    matches = re.findall(r'<article\b([^>]*data-report-entry[^>]*)>(.*?)</article>', source, re.S)
    if matches:
        result = []
        for attrs, body in matches:
            ident = re.search(r'\bid="([^"]+)"', attrs).group(1)
            group = re.search(r'\bdata-category="([^"]+)"', attrs).group(1)
            body = re.sub(r'^\s*<span class="tag">.*?</span>', '', body, count=1, flags=re.S)
            result.append(card(ident, group, body))
        # Other report writers may still append ordinary flat sections.
        result.extend(original_cards_legacy(re.sub(r'<section\b[^>]*data-catalog-panel.*?</section>', '', source, flags=re.S)))
        return result
    return original_cards_legacy(source)


def original_cards_legacy(source):
    result = []
    for attrs, body in re.findall(r'<section\b([^>]*)>(.*?)</section>', source, re.S):
        if '<h2' not in body or 'data-report-entry' in body:
            continue
        match = re.search(r'\bid="([^"]+)"', attrs)
        identifier = match.group(1) if match else 'entry-' + hashlib.sha256(body.encode()).hexdigest()[:12]
        result.append(card(identifier, category(unquote(body) + identifier), body))
    return result


def baseline_card():
    pointer = json.loads((ROOT / '.regression-control/baseline.json').read_text())
    directory = Path(pointer['path'])
    run = json.loads((directory / '运行记录.json').read_text())
    complete = run.get('complete') and run.get('integrity_verified') and (directory / '封存校验.json').is_file()
    rows = run['records']
    failures = sum(r['status'] != 'completed' for r in rows)
    finished = run.get('completed_at')
    date = datetime.strptime(finished, '%Y%m%d_%H%M%S_%f').strftime('%Y-%m-%d %H:%M') if finished else '尚未完成'
    modes = Counter(r['mode'] for r in rows)
    state = '已完成记录并封存' if complete else '记录尚未完整，不可验收'
    version = directory.name.rsplit('_', 1)[-1]
    overview = OUT / '完整基线效果总览_20261029_V1/去水印效果分档总览_20261029.html'
    entry = overview if overview.exists() else directory / '项目防退化控制报告.html'
    entry_label = '按效果分档查看' if overview.exists() else '三种模式逐张结果'
    body = f'<h2><a href="{href(entry)}">完整观察基线 {escape(version)}｜{entry_label}</a></h2><p class="state">{state} · {date} · 北京时间</p>'
    if overview.exists():
        body += '<p>新增三档效果总览：效果较好、一般／待复核、问题较多。可切换三种处理模式并查看实际原图与输出；分档仅为自动初筛。</p>'
        removal = ROOT / 'config/regression_retirements.json'
        if removal.exists():
            body += '<p>按用户要求移除3张手写样本后，当前效果页与后续回归为395个输入／423页／三模式1,269页次；下列1,278页次是原始封存范围。样本变更不代表算法效果提升。</p>'
    body += (f'<p>398个输入场景、426张／页，三种模式共 <strong>{len(rows):,}／{len(run["expected_keys"]):,} 页次</strong>。'
             f'运行或评分异常 {failures}；桌面默认 {modes["desktop"]}、关闭补全 {modes["desktop_protected"]}、CLI默认 {modes["cli"]} 页次。</p>'
             '<p><strong>观察基线记录当时表现，包括既有缺陷；不是当前工作版本全量通过、全部图片效果合格或独立泛化验证。</strong></p>')
    links = [('逐图报告', '项目防退化控制报告.html'), ('运行记录与指标', '运行记录.json'),
             ('完整样本清单', '原始清单.json'), ('冻结验收规则', '验收规则.json'), ('封存校验', '封存校验.json')]
    body += '<p class="links">' + ' · '.join(f'<a href="{href(directory / filename)}">{label}</a>' for label, filename in links) + '</p>'
    return card('complete-observation-baseline-20261028-v3', 'baseline', body), directory


def discover(cards, baseline):
    linked = set()
    for item in cards:
        parser = Text()
        parser.feed(item['body'])
        for link in parser.links:
            if not urlsplit(link).scheme:
                linked.add((OUT / unquote(urlsplit(link).path)).resolve())
    additions = []
    for path in sorted(OUT.rglob('*.html')):
        if path.resolve() in linked or path == INDEX:
            continue
        name = path.name
        if any(k in str(path) for k in ('/代码快照/', '/基线代码/', '/暂存代码快照/', '/可维护源文件/', '/原生编辑工作区/', '/样本移除留档/', '/整理前备份/', '/更新前备份/')):
            continue
        if any(k in name for k in ('入口片段', '备份', '留档', '修改前', '更新前', '添加方案入口前', '分类前')):
            continue
        if name in ('水印选区与处理结果复核.html', '官网列表.html'):
            continue
        if name == 'review.html' and any(k in str(path) for k in ('recall_priority', 'stroke_selection')):
            continue  # Batch drilldowns are reachable from their parent report.
        parser = Text()
        parser.feed(path.read_text(errors='replace'))
        title = ''.join(parser.title).strip()
        if not title or title == '水印验证报告总目录':
            continue
        relative = path.relative_to(OUT).as_posix()
        group = category(relative + title)
        if group == 'control' and '自检' in relative:
            title = '管控自检｜' + ('同版本对照' if '同版本' in relative else '冻结与三模式执行')
        if group == 'history' and title == '最小集回归结果':
            title = '历史固定回归｜' + path.parent.name
        identifier = 'report-' + hashlib.sha256(relative.encode()).hexdigest()[:12]
        description = ('仅验证工具链路，不作为去水印质量成绩。' if '自检' in relative
                       else '补充收录的已有报告。样本、参数、代码版本和结论以原报告为准，不能直接并入当前成绩。')
        body = (f'<h2><a href="{href(path)}">{escape(title)}</a></h2><p>{description}</p>'
                f'<p class="path">{escape(relative)}</p>')
        additions.append(card(identifier, group, body))
    return additions


CSS = '''
:root{--bg:#f3f5f3;--ink:#203a32;--muted:#5c7068;--line:#dce5df;--green:#176a4d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
.shell{max-width:1240px;margin:0 auto;padding:44px 32px 70px}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:650;color:var(--green)}
h1{font-size:34px;line-height:1.3;margin:12px 0}header>p{color:var(--muted);max-width:870px;margin:10px 0}.timestamp{font-size:13px;color:var(--muted)}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}.summary span{padding:6px 12px;background:white;border:1px solid var(--line);border-radius:7px;font-size:13px}
.tabs{display:flex;gap:5px;overflow-x:auto;border-bottom:1px solid var(--line);padding-bottom:1px}.tabs button{white-space:nowrap;min-height:48px;padding:11px 15px;border:0;border-bottom:3px solid transparent;background:transparent;color:var(--muted);font:inherit;cursor:pointer}
.tabs button[aria-selected=true]{border-bottom-color:var(--green);color:var(--green);font-weight:650}.count{font-size:12px;margin-left:7px;opacity:.8}
button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #5ba887;outline-offset:3px;border-radius:3px}
.toolbar{display:flex;justify-content:space-between;gap:22px;align-items:center;margin:25px 0 20px}.toolbar h2{font-size:20px;margin:0}.toolbar p{font-size:13px;color:var(--muted);margin:5px 0 0}
.search{display:flex;align-items:center;gap:8px;flex-shrink:0}.search input{width:280px;max-width:100%;border:1px solid #c7d5cc;border-radius:8px;padding:11px 12px;background:white;color:var(--ink);font:inherit}.search button{background:none;border:0;color:var(--green);font:inherit;cursor:pointer;padding:8px}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{min-width:0;background:white;padding:24px;border:1px solid var(--line);border-radius:12px}
.card[data-category=baseline]{grid-column:1/-1;border-top:3px solid var(--green);padding:28px}.card h2{font-size:18px;line-height:1.55;margin:0 0 9px}.card p{margin:9px 0;font-size:14px}.card a{color:var(--green);text-underline-offset:3px;text-decoration-thickness:1px}.card h2 a{text-decoration:none}.card h2 a:hover{text-decoration:underline}
.card .path{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.state{font-weight:650;color:var(--green)}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:1px 7px;font-size:11px;color:var(--muted);margin-bottom:12px}
.card a[aria-disabled=true]{color:var(--muted);text-decoration:none;cursor:default}.card small[data-missing-note]{color:#885c24}
[hidden]{display:none!important}.empty{padding:40px;background:white;border:1px dashed var(--line);border-radius:10px;text-align:center;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:32px;padding-top:16px;font-size:12px;color:var(--muted)}
@media(max-width:760px){.shell{padding:24px 16px 40px}h1{font-size:27px}.cards{grid-template-columns:1fr}.toolbar{flex-direction:column;align-items:stretch}.search{width:100%}.search input{width:100%}.search label{flex:1}.tabs button{padding:10px}.card{padding:19px}}
@media print{.tabs,.search{display:none}.card[hidden]{display:block!important}.cards{display:block}.card{break-inside:avoid;margin:12px 0}.shell{padding:0}}
'''

JS = '''
(() => {
  const buttons = [...document.querySelectorAll('[role="tab"]')];
  const cards = [...document.querySelectorAll('[data-report-entry]')];
  const search = document.querySelector('#report-search');
  const title = document.querySelector('#category-heading');
  const description = document.querySelector('#category-description');
  const panel = document.querySelector('#report-panel');
  const result = document.querySelector('#result-count');
  const empty = document.querySelector('#empty-state');
  let category = 'baseline';
  function filter(writeHash = true) {
    const button = buttons.find(item => item.dataset.category === category) || buttons[0];
    category = button.dataset.category;
    buttons.forEach(item => {
      const active = item === button;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
    });
    panel.setAttribute('aria-labelledby', button.id);
    title.textContent = button.dataset.title;
    description.textContent = button.dataset.description;
    const query = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    cards.forEach(card => {
      const match = (category === 'all' || card.dataset.category === category) &&
        (!query || card.textContent.toLocaleLowerCase().includes(query));
      card.hidden = !match;
      visible += match ? 1 : 0;
    });
    result.textContent = `显示 ${visible} 个入口`;
    empty.hidden = visible !== 0;
    if (writeHash) {
      const state = new URLSearchParams({tab: category});
      if (query) state.set('q', search.value);
      history.replaceState(null, '', '#' + state.toString());
    }
  }
  function readHash() {
    const hash = location.hash.slice(1);
    if (hash.includes('=')) {
      const state = new URLSearchParams(hash);
      category = state.get('tab') || 'baseline';
      search.value = state.get('q') || '';
    } else if (hash) {
      const target = document.getElementById(decodeURIComponent(hash));
      const card = target?.closest('[data-report-entry]');
      if (card) category = card.dataset.category;
    }
    filter(false);
    if (hash && !hash.includes('=')) document.getElementById(decodeURIComponent(hash))?.scrollIntoView();
  }
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => {category = button.dataset.category; filter();});
    button.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
      else if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = buttons.length - 1;
      else return;
      event.preventDefault();
      buttons[next].focus();
      buttons[next].click();
    });
  });
  search.addEventListener('input', () => filter());
  document.querySelector('#clear-search').addEventListener('click', () => {search.value = ''; filter(); search.focus();});
  window.addEventListener('hashchange', readHash);
  readHash();
})();
'''


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    source = INDEX.read_text()
    folder = OUT / '报告目录整理' / now.strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(INDEX, folder / '总目录_分类前备份.html')
    curated = original_cards(source)
    baseline, directory = baseline_card()
    report = directory / '项目防退化控制报告.html'
    sealed_report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    # Current full observation supersedes the old wording, not the old files.
    for item in curated:
        if '当前默认基线｜两批14张' in item['body']:
            item['body'] = item['body'].replace('当前默认基线｜两批14张', '历史开发基线｜两批14张')
    cards = [baseline] + [item for item in curated if item['id'] != baseline['id']]
    current_id = 'current-project-organization-20261029'
    current_body = ('<h2>' + '<a href="' + href(ROOT / 'docs/项目文档总览.html') + '">当前说明｜项目状态、数据与工具入口（2026-10-29）</a></h2>'
                    '<p>先看当前状态，再选择观察基线、实验或历史记录。数据数量从实际清单生成；本轮只整理入口和管理，不形成新的算法效果成绩。</p>'
                    '<p>' + '<a href="' + href(ROOT / 'docs/项目现状与后续开发说明.html') + '">当前状态与待处理事项</a> · '
                    '<a href="' + href(ROOT / 'docs/代码与工具用途说明.html') + '">代码、工具与备份</a></p>')
    cards = [item for item in cards if item['id'] != current_id] + [card(current_id, 'control', current_body)]
    failure_record = ROOT / 'docs/去水印失败案例记录.html'
    if failure_record.is_file() and 'id="baseline-review-20261029"' in failure_record.read_text():
        identifier = 'failure-baseline-review-20261029'
        body = ('<h2><a href="' + href(failure_record) + '">'
                '失败案例更新｜完整观察基线问题归纳与6组代表证据（2026-10-29）</a></h2>'
                '<p>保留原有11个历史案例，补充空选区、重建差异、内容损伤、模式双向影响和修复范围问题；'
                '区分已确认现象、实现风险与待验证原因。仅分析保存结果，未修改算法或重跑推理。</p>')
        cards = [item for item in cards if item['id'] != identifier] + [card(identifier, 'diagnosis', body)]
    for item in cards:
        item['body'] = available_local_links(clean_document_links(item['body']))
    cards += discover(cards, directory)
    unique = {item['id']: item for item in cards}
    cards = list(unique.values())
    for item in cards:
        item['body'] = available_local_links(clean_document_links(item['body']))
    counts = Counter(item['category'] for item in cards)
    tabs = ''.join(f'<button id="tab-{key}" type="button" role="tab" aria-controls="report-panel" aria-selected="{str(key == "baseline").lower()}" tabindex="{0 if key == "baseline" else -1}" data-category="{key}" data-title="{label}" data-description="{escape(description, quote=True)}">{label}<span class="count">{len(cards) if key == "all" else counts[key]}</span></button>' for key, (label, description) in CATEGORIES.items())
    body = []
    for item in cards:
        label = CATEGORIES[item['category']][0]
        body.append(f'<article id="{item["id"]}" class="card" data-report-entry="true" data-category="{item["category"]}"><span class="tag">{label}</span>{item["body"]}</article>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水印验证报告总目录</title><style>{CSS}</style></head>
<body><main class="shell"><header><div class="eyebrow">离线智能去水印 · 报告中心</div><h1>水印验证报告总目录</h1><p>按用途查看观察基线、项目管控、失败诊断、算法实验、历史回归与方案材料。保留每轮原始证据，历史结果和未通过实验分别标明。</p>
<!-- report-timestamp:start -->
<p class="timestamp" data-report-timestamp="updated">目录更新时间：<time datetime="{now.isoformat(timespec='seconds')}">{now.strftime('%Y-%m-%d %H:%M:%S')}</time> · 北京时间</p>
<!-- report-timestamp:end -->
<div class="summary"><span>{len(cards)} 个报告与材料入口</span><span>历史完整观察基线：1,278 页次已记录</span><span>记录完成 ≠ 当前代码验收通过</span></div><p><a href="{href(ROOT / 'docs/项目现状与后续开发说明.html')}">查看当前状态与清单问题</a> · <a href="{href(ROOT / 'docs/项目文档总览.html')}">查看文档用途</a></p></header>
<nav class="tabs" role="tablist" aria-label="报告分类">{tabs}</nav>
<section id="report-panel" data-catalog-panel="true" role="tabpanel" aria-labelledby="tab-baseline"><div class="toolbar"><div><h2 id="category-heading">全部报告</h2><p id="category-description">按上方标签切换分类。</p><p id="result-count" role="status" aria-live="polite">共 {len(cards)} 个入口</p></div><div class="search"><label><span class="sr-label" hidden>搜索当前分类的报告</span><input id="report-search" type="search" aria-label="搜索当前分类的报告" placeholder="在当前分类搜索报告…"></label><button id="clear-search" type="button">清空</button></div></div>
<div class="cards">{''.join(body)}</div><p id="empty-state" class="empty" hidden>当前分类没有匹配的报告。可清空搜索或切换“全部报告”。</p></section>
<noscript><p>JavaScript 未启用，以下已展示全部报告；浏览器查找仍可使用。</p></noscript><footer>本页与报告均从本地文件打开，无需联网。点击标题查看完整报告；子批次、逐图输出和原始数据从对应主报告继续进入。目录分类不改变原始报告、实验结论或封存记录。</footer></main><script>{JS}</script></body></html>'''
    INDEX.write_text(page)
    if hashlib.sha256(report.read_bytes()).hexdigest() != sealed_report_hash:
        raise RuntimeError('封存报告不应被目录整理修改')
    save = dict(updated_at=now.isoformat(), entries=len(cards), by_category=dict(counts),
                baseline_report_unchanged=True, backup=str(folder / '总目录_分类前备份.html'),
                preserved_original_entries=len(curated), added_entries=len(cards) - len(curated))
    (folder / '目录整理记录.json').write_text(json.dumps(save, ensure_ascii=False, indent=2))
    print(json.dumps(save, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
