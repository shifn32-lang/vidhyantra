"""Chat transcripts as a designed PDF or a tidy plain-text file.

Used by the account menu's "Export all chats" and the per-chat PDF download.
The callers hand over plain data (see ``Chat`` / ``Turn``); nothing here
touches the database, so it is easy to test.

The PDF is laid out as HTML/CSS through PyMuPDF's Story engine rather than
drawn glyph by glyph. That engine shapes complex scripts and carries its own
fallback fonts, so Hindi, Tamil, Bengali, Arabic, Chinese, Japanese, Korean
and the rest come out as real text instead of empty boxes.
"""
import html
import io
import re
import textwrap
from dataclasses import dataclass, field

from myapp.file_convert import ConvertError

TEXT_WIDTH = 78
_PAGE = (595, 842)           # A4, points
_MARGINS = (54, 62, 54, 58)  # left, top, right, bottom
_INK = '#1f2430'
_MUTED = '#6b7280'
_RULE = '#d9dce3'
_ACCENT = '#4338ca'
_ACCENT_SOFT = '#eef0ff'
_USER_SOFT = '#f3f4f6'


@dataclass
class Turn:
    role: str                 # 'user' or 'assistant'
    who: str                  # "You" or the model's name
    when: str                 # "03 Oct 2026, 11:40 PM"
    time: str                 # "11:40 PM"
    content: str = ''
    notes: list = field(default_factory=list)   # "[image in this message]" etc.


@dataclass
class Chat:
    title: str
    turns: list
    started: str = ''
    subtitle: str = ''        # e.g. "Started 03 Oct 2026 · 4 messages"


def _pluralize(count, word):
    return f'{count} {word}{"" if count == 1 else "s"}'


def _clip(text, limit):
    text = ' '.join((text or '').split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


# --------------------------------------------------------------------- text

def _indent(text, prefix='    '):
    return '\n'.join((prefix + line).rstrip() for line in text.splitlines())


def build_text(brand, chats, exported_at, note=''):
    """One UTF-8 plain-text document: a banner, a contents list, then every
    chat under a ruled heading with each message's speaker and time on its
    own line and the body indented beneath it."""
    heavy, light = '=' * TEXT_WIDTH, '-' * TEXT_WIDTH
    total = sum(len(chat.turns) for chat in chats)
    lines = [heavy, f'{brand} AI — all chats', heavy,
             f'Exported : {exported_at}',
             f'Chats    : {len(chats)}',
             f'Messages : {total}']
    if note:
        lines.append(f'Note     : {note}')
    lines += ['', 'CONTENTS', light]
    width = len(str(len(chats)))
    for number, chat in enumerate(chats, 1):
        lines.append(f'{number:>{width}}. {_clip(chat.title, 52):<52}  {chat.started}')
    lines.append('')

    for number, chat in enumerate(chats, 1):
        lines += ['', heavy, f'CHAT {number} of {len(chats)}', chat.title, chat.subtitle, heavy, '']
        previous_day = None
        for turn in chat.turns:
            day = turn.when.rsplit(',', 1)[0]
            if day != previous_day:
                lines += [f'  ── {day} ──', '']
                previous_day = day
            lines.append(f'[{turn.time}] {turn.who}')
            body = (turn.content or '').strip()
            if body:
                lines.append(_indent(body))
            for item in turn.notes:
                lines.append(f'    {item}')
            lines.append('')
    lines += [light, f'End of export · {_pluralize(len(chats), "chat")}', '']
    return '\n'.join(lines)


# --------------------------------------------------------------- markdown→html

_LINK = re.compile(r'!?\[([^\]]*)\]\(([^)\s]+)[^)]*\)')
_BOLD = re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1')
_ITALIC = re.compile(r'(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])')
_CODE = re.compile(r'`([^`\n]+)`')


def _inline(text):
    """Inline Markdown → HTML. Code spans are lifted out first so nothing
    inside them is re-interpreted."""
    spans = []

    def stash(match):
        spans.append(f'<code>{html.escape(match.group(1))}</code>')
        return f'\x00{len(spans) - 1}\x00'

    text = _CODE.sub(stash, text)

    def link(match):
        label, url = match.group(1), match.group(2)
        if match.group(0).startswith('!'):
            return html.escape(label or 'image')
        if not label or label == url:
            return f'<span class="url">{html.escape(url)}</span>'
        return f'{html.escape(label)} <span class="url">({html.escape(url)})</span>'

    pieces = []
    last = 0
    for match in _LINK.finditer(text):
        pieces.append(html.escape(text[last:match.start()]))
        pieces.append(link(match))
        last = match.end()
    pieces.append(html.escape(text[last:]))
    out = ''.join(pieces)
    # html.escape turned & < > into entities, which the patterns below skip.
    out = _BOLD.sub(r'<b>\2</b>', out)
    out = _ITALIC.sub(r'<i>\1</i>', out)
    return re.sub(r'\x00(\d+)\x00', lambda m: spans[int(m.group(1))], out)


_FENCE = re.compile(r'^\s*(```|~~~)')
_BULLET = re.compile(r'^(\s*)[-*+•]\s+(.*)$')
_NUMBER = re.compile(r'^(\s*)\d+[.)]\s+(.*)$')
_HEADING = re.compile(r'^\s*(#{1,6})\s+(.*?)\s*#*\s*$')
_RULE_LINE = re.compile(r'^\s*([-*_])(\s*\1){2,}\s*$')
_TABLE_SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')


def _cells(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _code_block(lines):
    wrapped = []
    for line in lines:
        line = line.expandtabs(4)
        wrapped.extend(textwrap.wrap(line, 84, replace_whitespace=False, drop_whitespace=False,
                                     break_long_words=True) or [''])
    return f'<pre>{html.escape(chr(10).join(wrapped))}</pre>'


def markdown_to_html(text):
    """The Markdown models actually produce: paragraphs, headings, bullet and
    numbered lists (nested by indent), quotes, tables, rules, fenced code."""
    lines = (text or '').replace('\r\n', '\n').replace('\r', '\n').split('\n')
    out, paragraph, i = [], [], 0
    list_stack = []   # [(indent, 'ul'|'ol')]

    def flush_paragraph():
        if paragraph:
            out.append('<p>' + '<br/>'.join(_inline(line) for line in paragraph) + '</p>')
            paragraph.clear()

    def close_lists(to_indent=-1):
        while list_stack and list_stack[-1][0] > to_indent:
            out.append(f'</li></{list_stack.pop()[1]}>')

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush_paragraph()
            close_lists()
            fence = _FENCE.match(line).group(1)
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                block.append(lines[i])
                i += 1
            out.append(_code_block(block))
            i += 1
            continue
        if not line.strip():
            flush_paragraph()
            close_lists()
            i += 1
            continue
        if _RULE_LINE.match(line):
            flush_paragraph()
            close_lists()
            out.append('<hr/>')
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            close_lists()
            level = min(len(heading.group(1)) + 2, 6)   # keep message headings below the chat title
            out.append(f'<h{level}>{_inline(heading.group(2))}</h{level}>')
            i += 1
            continue
        if '|' in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]) and '-' in lines[i + 1]:
            flush_paragraph()
            close_lists()
            head = _cells(line)
            rows = []
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            table = ['<table class="md"><tr>' + ''.join(f'<th>{_inline(c)}</th>' for c in head) + '</tr>']
            for row in rows:
                row += [''] * (len(head) - len(row))
                table.append('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in row[:len(head)]) + '</tr>')
            out.append(''.join(table) + '</table>')
            continue
        if line.lstrip().startswith('>'):
            flush_paragraph()
            close_lists()
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith('>'):
                quote.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            out.append('<blockquote>' + '<br/>'.join(_inline(q) for q in quote) + '</blockquote>')
            continue
        item = _BULLET.match(line) or _NUMBER.match(line)
        if item:
            flush_paragraph()
            kind = 'ul' if _BULLET.match(line) else 'ol'
            indent = len(item.group(1).expandtabs(4))
            if list_stack and indent < list_stack[-1][0]:
                close_lists(indent)
            if list_stack and indent == list_stack[-1][0]:
                out.append('</li>')
                if list_stack[-1][1] != kind:
                    out.append(f'</{list_stack.pop()[1]}>')
            if not list_stack or indent > list_stack[-1][0]:
                list_stack.append((indent, kind))
                out.append(f'<{kind}>')
            out.append(f'<li>{_inline(item.group(2))}')
            i += 1
            continue
        if list_stack and line.startswith((' ', '\t')):
            out.append('<br/>' + _inline(line.strip()))   # wrapped text of a list item
            i += 1
            continue
        close_lists()
        paragraph.append(line.strip())
        i += 1
    flush_paragraph()
    close_lists()
    return '\n'.join(out)


# ---------------------------------------------------------------------- pdf

_CSS = f"""
body {{ font-family: sans-serif; font-size: 10pt; color: {_INK}; line-height: 1.38; }}
h1, h2, h3, h4, h5, h6, p {{ margin: 0; }}
.cover {{ margin-bottom: 18pt; padding-bottom: 12pt; border-bottom: 2pt solid {_ACCENT}; }}
.cover .brand {{ font-size: 11pt; font-weight: bold; color: {_ACCENT}; letter-spacing: 1pt; }}
.cover h1 {{ font-size: 26pt; margin-top: 4pt; color: {_INK}; }}
.cover .meta {{ font-size: 9.5pt; color: {_MUTED}; margin-top: 6pt; }}
.contents-h {{ font-size: 13pt; font-weight: bold; margin: 0 0 6pt 0; color: {_INK}; }}
table.toc {{ width: 456pt; border-collapse: collapse; }}
table.toc td {{ padding: 3.5pt 0; border-bottom: 0.5pt solid {_RULE}; font-size: 9.5pt; }}
table.toc td.n {{ width: 26pt; color: {_MUTED}; }}
table.toc td.t {{ width: 290pt; }}
table.toc td.d {{ width: 100pt; color: {_MUTED}; }}
table.toc td.p {{ width: 40pt; text-align: right; color: {_MUTED}; }}
.chat-title {{ font-size: 18pt; font-weight: bold; color: {_INK}; line-height: 1.25; margin: 0 0 3pt 0; padding-left: 9pt; border-left: 4pt solid {_ACCENT}; }}
.chat-title.plain {{ font-weight: normal; }}
.chat-meta {{ font-size: 9pt; color: {_MUTED}; margin: 0 0 12pt 13pt; }}
.day {{ text-align: center; font-size: 8.5pt; color: {_MUTED}; margin: 10pt 0 6pt 0; }}
.turn {{ margin: 0 0 9pt 0; padding: 7pt 10pt 8pt 10pt; }}
.turn.user {{ background-color: {_USER_SOFT}; border-left: 3pt solid #9ca3af; }}
.turn.assistant {{ background-color: {_ACCENT_SOFT}; border-left: 3pt solid {_ACCENT}; }}
.who {{ font-size: 9.5pt; font-weight: bold; margin-bottom: 3pt; }}
.user .who {{ color: #374151; }}
.assistant .who {{ color: {_ACCENT}; }}
.when {{ font-weight: normal; color: {_MUTED}; font-size: 8.5pt; }}
.body p {{ margin: 0 0 4pt 0; }}
.body h3, .body h4, .body h5, .body h6 {{ margin: 6pt 0 3pt 0; font-weight: bold; }}
.body h3 {{ font-size: 11.5pt; }} .body h4 {{ font-size: 10.5pt; }} .body h5, .body h6 {{ font-size: 10pt; }}
.body ul, .body ol {{ margin: 0 0 4pt 0; padding-left: 16pt; }}
.body li {{ margin: 0 0 1.5pt 0; }}
.body blockquote {{ margin: 3pt 0 5pt 0; padding-left: 8pt; border-left: 2pt solid {_RULE}; color: #4b5563; }}
.body hr {{ border: 0; border-top: 0.5pt solid {_RULE}; margin: 6pt 0; }}
.body pre {{ font-family: monospace; font-size: 8.5pt; line-height: 1.3; background-color: #ffffff; border: 0.5pt solid {_RULE}; padding: 5pt 7pt; margin: 3pt 0 6pt 0; }}
.body code {{ font-family: monospace; font-size: 9pt; background-color: #ffffff; }}
.body table.md {{ border-collapse: collapse; margin: 3pt 0 6pt 0; }}
.body table.md th, .body table.md td {{ border: 0.5pt solid #c7cbd4; padding: 2.5pt 5pt; font-size: 9pt; text-align: left; }}
.body table.md th {{ background-color: #ffffff; font-weight: bold; }}
.url {{ color: {_ACCENT}; }}
.note {{ font-size: 8.5pt; color: {_MUTED}; margin-top: 3pt; }}
"""


def _toc_html(chats, pages):
    rows = []
    for number, chat in enumerate(chats, 1):
        page = pages.get(number)
        rows.append(
            f'<tr><td class="n">{number}</td><td class="t">{html.escape(_clip(chat.title, 48))}</td>'
            f'<td class="d">{html.escape(chat.started.rsplit(",", 1)[0])}</td>'
            f'<td class="p">{page if page else ""}</td></tr>'
        )
    return '<p class="contents-h">Contents</p><table class="toc">' + ''.join(rows) + '</table>'


def _is_latin(text):
    """False when the text holds non-Latin script. Those glyphs come from
    fallback fonts that only ship a regular weight; asking for bold makes the
    engine smear them, so headings in such titles stay regular."""
    return all(ord(ch) < 0x250 or not ch.isalpha() for ch in text)


def _wrap(body):
    return '<html><body>' + body + '</body></html>'


def _chat_html(number, chat):
    parts = [
        f'<h2 class="chat-title{"" if _is_latin(chat.title) else " plain"}" id="chat-{number}">'
        f'{html.escape(chat.title)}</h2>',
        f'<p class="chat-meta">{html.escape(chat.subtitle)}</p>',
    ]
    previous_day = None
    for turn in chat.turns:
        day = turn.when.rsplit(',', 1)[0]
        if day != previous_day:
            parts.append(f'<p class="day">— {html.escape(day)} —</p>')
            previous_day = day
        body = markdown_to_html(turn.content) if (turn.content or '').strip() else ''
        notes = ''.join(f'<p class="note">{html.escape(n)}</p>' for n in turn.notes)
        kind = 'user' if turn.role == 'user' else 'assistant'
        parts.append(
            f'<div class="turn {kind}"><div class="who">{html.escape(turn.who)} '
            f'<span class="when">· {html.escape(turn.time)}</span></div>'
            f'<div class="body">{body}</div>{notes}</div>'
        )
    return '\n'.join(parts)


def _render(fitz, parts):
    """Lay each HTML part out on its own run of A4 pages (so every chat starts
    on a fresh page without a forced page break, which smears the previous
    message's background onto the new page). Returns (pdf bytes,
    {chat number: 1-based page})."""
    width, height = _PAGE
    left, top, right, bottom = _MARGINS
    where = fitz.Rect(left, top, width - right, height - bottom)
    mediabox = fitz.Rect(0, 0, width, height)
    positions = {}
    pages_done = 0
    pages_here = 0

    def rectfn(rect_num, filled):
        nonlocal pages_here
        pages_here = rect_num + 1
        return mediabox, where, fitz.Identity

    def positionfn(position):
        match = re.fullmatch(r'chat-(\d+)', position.id or '')
        if match and position.open_close & 1:
            positions.setdefault(int(match.group(1)), pages_done + position.page_num)

    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    for part in parts:
        pages_here = 0
        story = fitz.Story(html=part, user_css=_CSS)
        story.write(writer, rectfn, positionfn)
        pages_done += pages_here
    writer.close()
    return buffer.getvalue(), positions


def _page_html(brand, label, number, count):
    return (
        f'<table style="width:100%;border-collapse:collapse;font-family:sans-serif;font-size:8pt;color:{_MUTED}">'
        f'<tr><td>{html.escape(brand)} AI · {html.escape(label)}</td>'
        f'<td style="text-align:right">Page {number} of {count}</td></tr></table>'
    )


def build_pdf(brand, chats, exported_at='', note='', single=False):
    """A styled, paginated PDF. ``single`` drops the cover and contents for a
    one-chat download. Every chat starts on a fresh page; the PDF also gets a
    bookmark outline (one entry per chat) for its sidebar."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise ConvertError('PDF generation is temporarily unavailable.') from exc

    total = sum(len(chat.turns) for chat in chats)
    sections = [_wrap(_chat_html(number, chat)) for number, chat in enumerate(chats, 1)]

    def compose(pages):
        parts = []
        if not single:
            meta = f'Exported {exported_at} · {_pluralize(len(chats), "chat")} · {_pluralize(total, "message")}'
            if note:
                meta += f' · {note}'
            parts.append(_wrap(
                '<div class="cover"><div class="brand">' + html.escape(brand.upper()) + ' AI</div>'
                '<h1>All chats</h1><p class="meta">' + html.escape(meta) + '</p></div>'
                + _toc_html(chats, pages)
            ))
        return parts + sections

    data, positions = _render(fitz, compose({}))
    if not single and len(chats) > 1:
        # Second pass with the real page numbers in the contents list; the
        # numbers are narrow and right-aligned so the layout doesn't shift.
        data, positions = _render(fitz, compose(positions))

    doc = fitz.open('pdf', data)
    count = doc.page_count
    label = 'Chat export' if not single else _clip(chats[0].title, 60)
    for index, page in enumerate(doc, 1):
        rect = fitz.Rect(_MARGINS[0], _PAGE[1] - _MARGINS[3] + 14, _PAGE[0] - _MARGINS[2], _PAGE[1] - 20)
        page.draw_line((rect.x0, rect.y0 - 6), (rect.x1, rect.y0 - 6), color=(0.85, 0.86, 0.89), width=0.5)
        page.insert_htmlbox(rect, _page_html(brand, label, index, count), css='body{margin:0}')
    outline = [[1, _clip(chat.title, 80), positions[n]] for n, chat in enumerate(chats, 1) if n in positions]
    if outline:
        try:
            doc.set_toc(outline)
        except (ValueError, RuntimeError):
            pass
    doc.set_metadata({'title': f'{brand} AI — chat export', 'author': brand, 'producer': f'{brand} AI'})
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out
