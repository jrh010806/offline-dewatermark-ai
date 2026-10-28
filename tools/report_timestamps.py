"""Add visible, timezone-explicit times without inventing historical generation dates."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo('Asia/Shanghai')
START = '<!-- report-timestamp:start -->'
END = '<!-- report-timestamp:end -->'
BLOCK = re.compile(re.escape(START) + r'.*?' + re.escape(END), re.DOTALL)


class BodyPosition(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.lines = source.splitlines(keepends=True)
        self.body = None
        self.first_visible = None
        self.in_head = False

    def handle_starttag(self, tag, attrs):
        line, column = self.getpos()
        offset = sum(map(len, self.lines[:line - 1])) + column
        if tag == 'head':
            self.in_head = True
        if tag == 'body':
            self.body = offset + len(self.get_starttag_text())
        if (self.first_visible is None and not self.in_head
                and tag not in {'html', 'head', 'body', 'meta', 'link', 'title', 'style', 'script', 'base'}):
            self.first_visible = offset

    def handle_endtag(self, tag):
        if tag == 'head':
            self.in_head = False


def stamp_html(source: str, when: datetime, kind: str = 'generated') -> str:
    if kind not in {'generated', 'updated', 'historical'}:
        raise ValueError('Unknown timestamp kind')
    if when.tzinfo is None:
        raise ValueError('A timezone-aware datetime is required')
    when = when.astimezone(TIMEZONE)
    label = {'generated': '生成时间', 'updated': '目录更新时间',
             'historical': '历史报告时间'}[kind]
    note = '（依据补标前的文件修改时间）' if kind == 'historical' else ''
    block = (f'{START}\n<div data-report-timestamp="{kind}" '
             'style="box-sizing:border-box;max-width:1200px;margin:14px auto;padding:9px 14px;'
             'border:1px solid #d6e2db;border-radius:7px;background:#f3f7f4;color:#395348;'
             'font:14px/1.65 system-ui,sans-serif;overflow-wrap:anywhere">'
             f'{label}：<time datetime="{escape(when.isoformat(timespec="seconds"), quote=True)}">'
             f'{when:%Y-%m-%d %H:%M:%S}</time> {note} · 北京时间 UTC+08:00</div>\n{END}')
    if BLOCK.search(source):
        return BLOCK.sub(lambda _: block, source, count=1)
    parser = BodyPosition(source)
    parser.feed(source)
    position = parser.body if parser.body is not None else parser.first_visible
    if position is None:
        raise ValueError('Could not locate report body')
    return source[:position] + '\n' + block + '\n' + source[position:]


def stamp_existing(root: Path, audit_path: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob('*.html')):
        source = path.read_text(encoding='utf-8')
        if START in source:
            records.append({'path': str(path.resolve()), 'status': 'already_stamped'})
            continue
        original_mtime = path.stat().st_mtime
        when = datetime.fromtimestamp(original_mtime, TIMEZONE)
        path.write_text(stamp_html(source, when, 'historical'), encoding='utf-8')
        records.append({'path': str(path.resolve()), 'status': 'stamped',
                        'time_basis': 'file_mtime_before_timestamp_annotation',
                        'timestamp': when.isoformat(timespec='seconds')})
    import json
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({'annotation_time': datetime.now(TIMEZONE).isoformat(timespec='seconds'),
                                     'timezone': 'Asia/Shanghai', 'reports': records},
                                    ensure_ascii=False, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('out'))
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    records = stamp_existing(args.root, args.audit)
    print(f"HTML reports: {len(records)}; timestamps added: {sum(r['status'] == 'stamped' for r in records)}")
