"""The shared groundwork for every generated document (PDF, Word, PowerPoint,
Excel, plain text): a small Markdown block parser, an inline-formatting
tokenizer, number/table helpers and the colour palette.

Nothing here imports a document library, so the parser is easy to test on its
own and every writer in doc_render.py reads exactly the same structure. A
model writes Markdown; this turns it into blocks; each writer decides how a
heading, a table, a callout or a chart should look in its own format.
"""
import csv
import io
import re
from dataclasses import dataclass, field

# Six-digit hex, no '#': python-docx and python-pptx want it bare and the CSS
# writer adds the '#'. Emerald is the product's own accent colour.
PALETTE = {
    'ink': '111827', 'body': '1f2937', 'muted': '6b7280', 'faint': '9ca3af',
    'accent': '059669', 'accent_dark': '047857', 'deep': '064e3b',
    'mint': 'a7f3d0', 'soft': 'ecfdf5', 'zebra': 'f3f8f6',
    'rule': 'd1d5db', 'code_bg': 'f3f4f6', 'white': 'ffffff',
}

# kind -> (stripe colour, background, label)
CALLOUTS = {
    'note': ('2563eb', 'eff6ff', 'Note'),
    'tip': ('059669', 'ecfdf5', 'Tip'),
    'summary': ('059669', 'ecfdf5', 'Summary'),
    'warning': ('d97706', 'fffbeb', 'Warning'),
    'important': ('dc2626', 'fef2f2', 'Important'),
}
_CALLOUT_WORDS = {
    'note': 'note', 'info': 'note', 'information': 'note', 'remember': 'note',
    'tip': 'tip', 'hint': 'tip', 'pro tip': 'tip', 'key point': 'tip',
    'key points': 'tip', 'key takeaway': 'tip', 'key takeaways': 'tip',
    'takeaway': 'tip', 'example': 'tip',
    'summary': 'summary', 'executive summary': 'summary',
    'warning': 'warning', 'caution': 'warning', 'careful': 'warning',
    'important': 'important', 'critical': 'important', 'danger': 'important',
    'disclaimer': 'important',
}

CHART_COLORS = ['059669', '2563eb', 'f59e0b', '7c3aed', 'e11d48', '0891b2', '65a30d', 'db2777']
CHART_TYPES = {
    'column': 'column', 'bar': 'column', 'vertical': 'column', 'columns': 'column',
    'hbar': 'hbar', 'barh': 'hbar', 'horizontal': 'hbar', 'horizontal bar': 'hbar',
    'line': 'line', 'area': 'line', 'trend': 'line',
    'pie': 'pie', 'donut': 'donut', 'doughnut': 'donut',
}
MAX_CHART_POINTS = 24
MAX_CHART_SERIES = 6


def stamp(created=None):
    """``created`` (or now) as an aware datetime in Indian Standard Time — the
    product's own clock, whatever the server's time zone is."""
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        ist = ZoneInfo('Asia/Kolkata')
    except Exception:   # no tz database installed: IST has no daylight saving
        ist = timezone(timedelta(hours=5, minutes=30))
    moment = created or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(ist)


def date_text(created=None):
    """'6 October 2026'."""
    moment = stamp(created)
    return f'{moment.day} {moment.strftime("%B %Y")}'


@dataclass
class Block:
    """One structural piece of a document.

    kind is one of: heading, para, item, table, code, quote, rule, chart.
    ``item`` is a single list entry; ``level`` is how deeply it is nested.
    """
    kind: str
    text: str = ''
    level: int = 0
    ordered: bool = False
    number: int = 0
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    aligns: list = field(default_factory=list)
    lang: str = ''
    callout: str = ''
    chart: dict = None


# ──────────────────────────────────────────────────────────── inline text ──

@dataclass
class Span:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    strike: bool = False
    url: str = ''


_ESCAPABLE = '\\`*_{}[]()#+-.!|~<>'
_ESC_BASE = 0xE000   # private-use area: a stand-in that no pattern below matches
_INLINE = [
    ('code', re.compile(r'`([^`\n]+)`')),
    ('image', re.compile(r'!\[([^\]\n]*)\]\(\s*[^)\s]*[^)]*\)')),
    ('link', re.compile(r'\[([^\]\n]+)\]\(\s*([^)\s]+)[^)]*\)')),
    ('bold', re.compile(r'\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\*)|__(?=\S)(.+?)(?<=\S)__(?!_)', re.S)),
    ('strike', re.compile(r'~~(?=\S)(.+?)(?<=\S)~~')),
    ('italic', re.compile(
        r'(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])'
        r'|(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])')),
    ('url', re.compile(r'https?://[^\s<>"\']+')),
]
_BR_RE = re.compile(r'<br\s*/?>', re.I)
_TAG_RE = re.compile(r'</?(?:b|i|u|em|strong|span|div|p|small|sub|sup|mark)\b[^>]*>', re.I)


def _protect(text):
    return re.sub(
        r'\\([' + re.escape(_ESCAPABLE) + r'])',
        lambda m: chr(_ESC_BASE + ord(m.group(1))), text,
    )


_ESC_RANGE = re.compile('[' + chr(_ESC_BASE) + '-' + chr(_ESC_BASE + 0x7F) + ']')


def _restore(text):
    return _ESC_RANGE.sub(lambda m: chr(ord(m.group(0)) - _ESC_BASE), text)


def _parse_inline(text, fmt, out):
    pos = 0
    while pos < len(text):
        best = None
        for name, rx in _INLINE:
            match = rx.search(text, pos)
            if match and (best is None or match.start() < best[1].start()):
                best = (name, match)
        if best is None:
            out.append(Span(text[pos:], **fmt))
            return
        name, match = best
        if match.start() > pos:
            out.append(Span(text[pos:match.start()], **fmt))
        end = match.end()
        if name == 'code':
            out.append(Span(match.group(1), **{**fmt, 'code': True}))
        elif name == 'image':
            out.append(Span(match.group(1) or 'image', **fmt))
        elif name == 'link':
            url = match.group(2)
            if re.match(r'(?:https?:|mailto:|tel:)', url, re.I):
                _parse_inline(match.group(1), {**fmt, 'url': url}, out)
            else:
                _parse_inline(match.group(1), fmt, out)
        elif name == 'bold':
            _parse_inline(match.group(1) or match.group(2), {**fmt, 'bold': True}, out)
        elif name == 'strike':
            _parse_inline(match.group(1), {**fmt, 'strike': True}, out)
        elif name == 'italic':
            _parse_inline(match.group(1) or match.group(2), {**fmt, 'italic': True}, out)
        else:   # a bare URL: the sentence's own punctuation is not part of it
            url = match.group(0).rstrip('.,;:!?)]}\'"')
            out.append(Span(url, **{**fmt, 'url': fmt.get('url') or url}))
            end = match.start() + len(url)
        pos = end


def inline_spans(text):
    """Inline Markdown -> a list of formatted spans.

    Handles **bold**, *italic*, `code`, ~~strike~~, [label](url), bare URLs,
    backslash escapes and <br>. Images become their alt text — nothing is ever
    fetched from the network.
    """
    text = _protect(text or '')   # escapes first, so an escaped <br> stays text
    text = _BR_RE.sub('\n', text)
    text = _TAG_RE.sub('', text)
    out = []
    _parse_inline(text, {}, out)
    merged = []
    for span in out:
        span.text = _restore(span.text)
        if not span.text:
            continue
        if merged and all(
            getattr(merged[-1], key) == getattr(span, key)
            for key in ('bold', 'italic', 'code', 'strike', 'url')
        ):
            merged[-1].text += span.text
        else:
            merged.append(span)
    return merged


# Control characters are not allowed in Word, PowerPoint or Excel XML — a single
# stray one (a form feed or a NUL copied out of a PDF) makes the library refuse
# the whole file — so they are dropped before anything is written.
_XML_ILLEGAL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')


def clean_text(text):
    return _XML_ILLEGAL.sub('', text or '')


def escape_markdown(text):
    """Make raw data (a CSV cell, a file name) safe to hand to inline_spans:
    every character that could read as formatting is backslash-escaped."""
    return re.sub(r'([\\`*_{}\[\]()#+!|~<>])', r'\\\1', clean_text(text))


def host_of(url):
    """'https://www.example.com/a' -> 'example.com'; '' for anything that is not http(s)."""
    from urllib.parse import urlparse
    try:
        parts = urlparse(url)
    except ValueError:
        return ''
    if parts.scheme not in ('http', 'https'):
        return ''
    return (parts.hostname or '').removeprefix('www.')


def plain_text(text):
    """The words only: every Markdown marker resolved away."""
    return ''.join(span.text for span in inline_spans(text))


# ───────────────────────────────────────────────────────────────── numbers ──

_NUMBER_RE = re.compile(
    r'^\(?\s*[-+−–]?\s*(?:₹|Rs\.?|INR|USD|\$|€|£|¥)?\s*[-+−–]?\s*'
    r'\d[\d,]*(?:\.\d+)?\s*(?:%|k|m|bn|b|lakhs?|crores?|cr|L)?\s*\)?$',
    re.I,
)
_SUFFIX_FACTOR = {'k': 1e3, 'm': 1e6, 'b': 1e9, 'bn': 1e9, 'lakh': 1e5, 'lakhs': 1e5, 'crore': 1e7, 'crores': 1e7, 'cr': 1e7, 'l': 1e5}


def is_numeric(text):
    """True for a cell that reads as a quantity. A bare run of ten or more
    digits is an identifier (phone, account or order number), not a quantity, so
    its column stays left-aligned like the other text."""
    text = (text or '').strip()
    if not text or re.fullmatch(r'\d{10,}', text):
        return False
    return bool(_NUMBER_RE.match(text))


def parse_number(text):
    """A cell like '₹1,20,000', '12.5%', '(45)' or '3.2k' as a float, else None.

    Suffixes such as 'k' or 'lakh' are NOT multiplied out: a chart whose
    rows are all '12 lakh' / '15 lakh' should plot 12 and 15, not 1,200,000.
    """
    raw = (text or '').strip()
    if not raw or not _NUMBER_RE.match(raw):
        return None
    negative = raw.startswith('(') and raw.endswith(')')
    digits = re.search(r'\d[\d,]*(?:\.\d+)?', raw)
    if not digits:
        return None
    value = float(digits.group(0).replace(',', ''))
    before = raw[:digits.start()]
    if negative or re.search(r'[-−–]', before):
        value = -value
    return value


# ──────────────────────────────────────────────────────────────── parsing ──

_FENCE_RE = re.compile(r'^\s*(`{3,}|~{3,})\s*([^\s`]*)')
_HEADING_RE = re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$')
_BULLET_RE = re.compile(r'^(\s*)[-*+•◦▪‣◆●➤]\s+(.*\S.*)$')
_NUMBER_ITEM_RE = re.compile(r'^(\s*)(\d{1,4})[.)]\s+(.*\S.*)$')
_RULE_RE = re.compile(r'^\s*([-*_])(\s*\1){2,}\s*$')
_SETEXT_NOISE_RE = re.compile(r'^\s*={3,}\s*$')
_TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$')
_CALLOUT_RE = re.compile(
    r'^\s*(?:\[!(\w+)\]|\**\s*([A-Za-z][A-Za-z ]{1,20}?)\s*\**\s*[:：]\s*\**)\s*(.*)$', re.S,
)


def split_cells(line):
    """A '| a | b |' row -> ['a', 'b'], honouring \\| escapes."""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|') and not line.endswith('\\|'):
        line = line[:-1]
    return [cell.strip().replace('\\|', '|') for cell in re.split(r'(?<!\\)\|', line)]


def _alignments(sep_line, count):
    aligns = []
    for cell in split_cells(sep_line):
        cell = cell.strip()
        if cell.startswith(':') and cell.endswith(':'):
            aligns.append('center')
        elif cell.endswith(':'):
            aligns.append('right')
        elif cell.startswith(':'):
            aligns.append('left')
        else:
            aligns.append('')
    return (aligns + [''] * count)[:count]


def _callout_of(lines):
    """('tip', 'rest of the text') for '> **Tip:** ...' or '> [!TIP]', else ('', text)."""
    text = '\n'.join(lines).strip()
    match = _CALLOUT_RE.match(text)
    if match:
        word = (match.group(1) or match.group(2) or '').strip().lower()
        kind = _CALLOUT_WORDS.get(word)
        if kind:
            return kind, (match.group(3) or '').strip()
    return '', text


def parse_markdown(text):
    """Markdown the way models write it -> a flat list of Blocks.

    Consecutive plain lines stay one paragraph with their line breaks kept (a
    converted .txt file keeps its layout); a blank line starts a new paragraph.
    """
    lines = clean_text(text).replace('\r\n', '\n').replace('\r', '\n').split('\n')
    blocks = []
    para = []
    stack = []   # open list levels: [indent, ordered, counter]
    i = 0
    count = len(lines)

    def flush():
        if para:
            blocks.append(Block('para', text='\n'.join(para)))
            para.clear()

    def end_list():
        stack.clear()

    while i < count:
        line = lines[i]
        stripped = line.strip()

        fence = _FENCE_RE.match(line)
        if fence:
            flush()
            end_list()
            marker, lang = fence.group(1), fence.group(2).lower()
            body = []
            i += 1
            while i < count and not re.match(r'^\s*' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}\s*$', lines[i]):
                body.append(lines[i])
                i += 1
            i += 1
            code = '\n'.join(body).rstrip('\n')
            if lang == 'chart':
                spec = parse_chart('\n'.join(body))
                if spec:
                    blocks.append(Block('chart', chart=spec))
                    continue
            blocks.append(Block('code', text=code, lang=lang))
            continue

        if not stripped:
            flush()
            i += 1
            continue

        if _SETEXT_NOISE_RE.match(line):
            i += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            end_list()
            blocks.append(Block('heading', text=heading.group(2).strip(), level=len(heading.group(1))))
            i += 1
            continue

        if _RULE_RE.match(line) and '|' not in line:
            flush()
            end_list()
            blocks.append(Block('rule'))
            i += 1
            continue

        # A table is a '|' header row followed by a '---|---' separator row; a
        # run of '|…|' lines with no separator is still treated as one.
        sep_next = i + 1 < count and _TABLE_SEP_RE.match(lines[i + 1]) and '-' in lines[i + 1] and '|' in line
        piped_run = (
            stripped.startswith('|') and stripped.endswith('|') and i + 1 < count
            and lines[i + 1].strip().startswith('|') and lines[i + 1].strip().endswith('|')
        )
        if sep_next or piped_run:
            flush()
            end_list()
            header = split_cells(line)
            aligns = _alignments(lines[i + 1], len(header)) if sep_next else [''] * len(header)
            i += 2 if sep_next else 1
            rows = []
            while i < count and lines[i].strip() and '|' in lines[i]:
                if _TABLE_SEP_RE.match(lines[i]) and '-' in lines[i]:
                    i += 1
                    continue
                cells = split_cells(lines[i])
                rows.append((cells + [''] * len(header))[:len(header)])
                i += 1
            if not sep_next and not rows:
                para.append(stripped)
                continue
            blocks.append(Block('table', header=header, rows=rows, aligns=aligns))
            continue

        if stripped.startswith('>'):
            flush()
            end_list()
            quote = []
            while i < count and lines[i].lstrip().startswith('>'):
                quote.append(re.sub(r'^\s*>\s?', '', lines[i]))
                i += 1
            kind, body = _callout_of(quote)
            blocks.append(Block('quote', text=body, callout=kind))
            continue

        item = _BULLET_RE.match(line)
        ordered = False
        if not item:
            item = _NUMBER_ITEM_RE.match(line)
            ordered = bool(item)
        if item:
            flush()
            indent = len(item.group(1).expandtabs(4))
            body = item.group(len(item.groups()))
            while stack and indent < stack[-1][0]:
                stack.pop()
            if stack and indent == stack[-1][0] and stack[-1][1] != ordered:
                stack.pop()
            if not stack or indent > stack[-1][0]:
                stack.append([indent, ordered, int(item.group(2)) - 1 if ordered else 0])
            frame = stack[-1]
            if ordered:
                frame[2] += 1
            blocks.append(Block(
                'item', text=body.strip(), level=len(stack) - 1,
                ordered=ordered, number=frame[2] if ordered else 0,
            ))
            i += 1
            continue

        # Indented text right under a list item belongs to that item.
        indent = len(line) - len(line.lstrip())
        if stack and blocks and blocks[-1].kind == 'item' and indent > stack[-1][0]:
            blocks[-1].text += '\n' + stripped
            i += 1
            continue

        end_list()
        para.append(stripped)
        i += 1

    flush()
    return blocks


def document_title(blocks):
    """The text of a leading '# Title', or '' when the document has none."""
    for block in blocks:
        if block.kind == 'rule':
            continue
        if block.kind == 'heading' and block.level == 1:
            return plain_text(block.text)
        break
    return ''


# ─────────────────────────────────────────────────────────────────── tables ──

def column_aligns(block):
    """Per-column 'left' / 'right' / 'center', explicit or inferred.

    A column whose every filled cell is a number (₹ and % included) is
    right-aligned so the digits line up.
    """
    result = []
    for index in range(len(block.header)):
        explicit = block.aligns[index] if index < len(block.aligns) else ''
        if explicit:
            result.append(explicit)
            continue
        cells = [row[index] for row in block.rows if index < len(row) and row[index].strip()]
        result.append('right' if cells and all(is_numeric(cell) for cell in cells) else 'left')
    return result


def column_weights(block, floor=6, ceiling=48):
    """Relative column widths from the widest content, clamped so one long
    paragraph cell cannot squeeze the others to nothing."""
    weights = []
    for index, head in enumerate(block.header):
        sizes = [len(plain_text(head))] + [
            len(plain_text(row[index])) for row in block.rows if index < len(row)
        ]
        longest = max(sizes) if sizes else floor
        # The 80th percentile keeps a single outlier from setting the width.
        ordered = sorted(sizes)
        typical = ordered[min(len(ordered) - 1, int(len(ordered) * 0.8))]
        weights.append(max(floor, min(ceiling, max(typical, min(longest, 24)))))
    return weights


# ─────────────────────────────────────────────────────────────────── charts ──

_CHART_META_RE = re.compile(r'^\s*(type|title|unit|subtitle)\s*:\s*(.*)$', re.I)


def parse_chart(text):
    """The body of a ```chart block -> a plain dict, or None when unusable.

    type: column | hbar | line | pie | donut        (default column)
    title: Monthly revenue
    unit: ₹ lakh
    Month, Revenue, Cost          <- header: label column, then one per series
    Jan, 12.5, 9
    Feb, 14.1, 10
    """
    meta = {}
    rows = []
    for line in (text or '').splitlines():
        if not line.strip():
            continue
        match = _CHART_META_RE.match(line)
        if match and not rows:
            meta[match.group(1).lower()] = match.group(2).strip()
            continue
        try:
            rows.append(next(csv.reader(io.StringIO(line), skipinitialspace=True)))
        except (csv.Error, StopIteration):
            continue
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    header, body = rows[0], rows[1:MAX_CHART_POINTS + 1]
    series = []
    for column in range(1, min(len(header), MAX_CHART_SERIES + 1)):
        values = [
            parse_number(row[column]) if column < len(row) else None for row in body
        ]
        if any(value is not None for value in values):
            series.append({'name': header[column].strip() or f'Series {column}', 'values': values})
    if not series:
        return None
    kind = CHART_TYPES.get(meta.get('type', 'column').strip().lower(), 'column')
    if kind in ('pie', 'donut'):
        series = series[:1]
        if any(value is not None and value < 0 for value in series[0]['values']):
            kind = 'column'
        elif not any(value for value in series[0]['values']):
            return None
    return {
        'type': kind,
        'title': meta.get('title', ''),
        'subtitle': meta.get('subtitle', ''),
        'unit': meta.get('unit', ''),
        'labels': [(row[0] if row else '').strip() for row in body],
        'series': series,
    }


def format_value(value, places=None):
    """12500.0 -> '12,500', 12.5 -> '12.5', None -> ''."""
    if value is None:
        return ''
    if places is None:
        places = 0 if abs(value - round(value)) < 1e-9 else (1 if abs(value) >= 100 else 2)
    text = f'{value:,.{places}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text
