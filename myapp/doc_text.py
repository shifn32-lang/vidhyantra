"""Plain-text files that read well in Notepad, and small finishing touches for
other generated text files (CSV, JSON).

A .txt the model fills with Markdown shows literal ``**stars**``, ``## hashes``
and ``| --- |`` rows. ``render_txt`` turns that into text laid out for a
monospace page: a banner title, underlined headings, aligned tables, wrapped
paragraphs, bulleted lists and bar charts drawn with block characters. Text
that is not Markdown at all is left exactly as written.
"""
import json
import re
import textwrap
import unicodedata

from myapp.doc_blocks import (
    CALLOUTS, column_aligns, date_text, inline_spans, parse_markdown, plain_text,
)

WIDTH = 80
TABLE_MAX_WIDTH = 100
_MARKDOWN_HINT = re.compile(
    r'^\s{0,3}#{1,6}\s+\S'                         # a heading
    r'|^\s*\|?\s*:?-{3,}:?\s*\|[\s|:-]*$'          # a table separator row
    r'|^\s*(?:`{3,}|~{3,})'                        # a code fence
    r'|\*\*[^*\n]+\*\*'                            # **bold**
    r'|\[[^\]\n]+\]\(https?://',                   # [label](url)
    re.M,
)
_BULLETS = ('•', '◦', '▪')


def looks_like_markdown(text):
    return bool(_MARKDOWN_HINT.search(text or ''))


def _width(text):
    """Columns the text occupies: wide (CJK) characters count twice, combining
    marks (Hindi vowel signs and the like) not at all."""
    total = 0
    for ch in text:
        if unicodedata.category(ch) in ('Mn', 'Me', 'Cf'):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return total


def _pad(text, width, align='left'):
    gap = max(0, width - _width(text))
    if align == 'right':
        return ' ' * gap + text
    if align == 'center':
        return ' ' * (gap // 2) + text + ' ' * (gap - gap // 2)
    return text + ' ' * gap


def _readable(text):
    """Inline Markdown as plain words; a link keeps its address in brackets."""
    pieces = []
    for span in inline_spans(text):
        if span.url and span.text.strip() != span.url:
            pieces.append(f'{span.text} ({span.url})')
        else:
            pieces.append(span.text)
    return ''.join(pieces)


def _wrap(text, indent='', hanging=None):
    hanging = indent if hanging is None else hanging
    lines = []
    for raw in text.split('\n'):
        wrapped = textwrap.wrap(
            raw, width=WIDTH, initial_indent=indent if not lines else hanging,
            subsequent_indent=hanging, break_long_words=False, break_on_hyphens=False,
        )
        lines.extend(wrapped or [indent if not lines else ''])
    return lines


def _table(block):
    header = [_readable(cell) for cell in block.header]
    rows = [[_readable(cell) for cell in row] for row in block.rows]
    aligns = column_aligns(block)
    columns = len(header)
    widths = [max([_width(header[i])] + [_width(row[i]) for row in rows]) for i in range(columns)]
    overhead = 3 * columns + 1
    if sum(widths) + overhead > TABLE_MAX_WIDTH:
        # Too wide for a text page: each row becomes a small "Label: value" card.
        lines = []
        label = max(_width(h) for h in header)
        for number, row in enumerate(rows, 1):
            lines.append(f'Row {number}')
            for name, value in zip(header, row):
                lines.append(f'  {_pad(name, label)} : {value}')
            lines.append('')
        return lines[:-1]
    rule = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def line(cells, force_left=False):
        return '| ' + ' | '.join(
            _pad(cell, widths[i], 'left' if force_left else aligns[i]) for i, cell in enumerate(cells)
        ) + ' |'

    lines = [rule, line(header, force_left=False), rule]
    lines.extend(line(row) for row in rows)
    lines.append(rule)
    return lines


def _chart(spec):
    labels = spec['labels']
    label_width = min(24, max((_width(label) for label in labels), default=0))
    peak = max((abs(v) for s in spec['series'] for v in s['values'] if v is not None), default=0) or 1
    lines = [spec['title'] + (f' ({spec["unit"]})' if spec['unit'] else '')] if spec['title'] else []
    total = sum(v for v in spec['series'][0]['values'] if v) if spec['type'] in ('pie', 'donut') else 0
    for series in spec['series']:
        if len(spec['series']) > 1:
            lines.append(f'  {series["name"]}')
        for label, value in zip(labels, series['values']):
            if value is None:
                bar, shown = '', 'n/a'
            else:
                bar = ('█' if value >= 0 else '▒') * max(1, round(abs(value) / peak * 30)) if value else ''
                shown = f'{value:,.2f}'.rstrip('0').rstrip('.')
                if total:
                    shown += f'  ({value / total * 100:.1f}%)'
            lines.append(f'  {_pad(label[:24], label_width)}  {bar} {shown}'.rstrip())
    return lines


def render_txt(content, *, brand='', created=None, title=''):
    """Markdown-ish text -> neatly laid out plain text."""
    blocks = parse_markdown(content)
    out = []

    def gap():
        if out and out[-1] != '':
            out.append('')

    heavy, light = '=' * WIDTH, '-' * WIDTH
    has_title = False
    for index, block in enumerate(blocks):
        kind = block.kind
        if kind == 'heading':
            text = _readable(block.text).replace('\n', ' ')
            gap()
            if block.level == 1 and index == 0:
                has_title = True
                out.extend([heavy, text.upper(), heavy, f'Prepared on {date_text(created)}', ''])
                continue
            if block.level == 1:
                out.extend([text.upper(), '=' * min(WIDTH, max(_width(text), 8))])
            elif block.level == 2:
                out.extend([text.upper(), '-' * min(WIDTH, max(_width(text), 8))])
            elif block.level == 3:
                out.append(f'▸ {text}')
            else:
                out.append(f'{text}:')
            out.append('')
        elif kind == 'para':
            out.extend(_wrap(_readable(block.text)))
            out.append('')
        elif kind == 'item':
            text = _readable(block.text).replace('\n', ' ')
            marker = f'{block.number}.' if block.ordered else _BULLETS[min(block.level, 2)]
            indent = '  ' * block.level
            lead = f'{indent}{marker} '
            if out and out[-1] == '' and index and blocks[index - 1].kind == 'item':
                out.pop()
            out.extend(_wrap(text, lead, ' ' * len(lead)))
            if index + 1 >= len(blocks) or blocks[index + 1].kind != 'item':
                out.append('')
        elif kind == 'table':
            gap()
            out.extend(_table(block))
            out.append('')
        elif kind == 'quote':
            gap()
            if block.callout:
                tag = f'[{CALLOUTS[block.callout][2].upper()}] '
                out.extend(_wrap(_readable(block.text), tag, ' ' * len(tag)))
            else:
                out.extend(_wrap(_readable(block.text), '> ', '> '))
            out.append('')
        elif kind == 'code':
            gap()
            out.extend(('    ' + line).rstrip() for line in block.text.expandtabs(4).split('\n'))
            out.append('')
        elif kind == 'rule':
            gap()
            out.extend([light, ''])
        elif kind == 'chart':
            gap()
            out.extend(_chart(block.chart))
            out.append('')
    while out and out[-1] == '':
        out.pop()
    if has_title:
        out.extend(['', light, f'{brand} AI · {date_text(created)}' if brand else date_text(created)])
    return '\n'.join(out) + '\n'


def finish_text_file(file_name, content, *, brand='', created=None):
    """Bytes for a generated text-like file, with the small improvements that
    never change what it says."""
    extension = file_name.rsplit('.', 1)[-1].lower() if '.' in file_name else ''
    text = content or ''
    if extension == 'txt' and looks_like_markdown(text):
        text = render_txt(text, brand=brand, created=created)
    elif extension == 'json':
        # Only a minified one-liner is re-indented; a file already laid out
        # (or one whose repeated keys would be merged) is left exactly alone.
        if len(text.strip().splitlines()) <= 2:
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False) + '\n'
            except ValueError:
                pass
    elif extension == 'csv' and any(ord(ch) > 127 for ch in text):
        # A byte-order mark is how Excel knows a CSV is UTF-8 — without it ₹
        # and Hindi show up as garbage.
        return text.encode('utf-8-sig')
    return text.encode('utf-8')
