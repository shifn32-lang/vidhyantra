"""Designed PDFs from Markdown-ish text.

The page is laid out as HTML/CSS through PyMuPDF's Story engine (the same one
chat_export.py uses), which shapes complex scripts and carries its own
fallback fonts — Hindi, Tamil, Bengali, Kannada and Gujarati come out as real
text, not empty boxes. On top of that: a title band, styled tables, callouts,
code blocks, charts, clickable links, a contents list for long documents,
page footers and a bookmark outline.

Two quirks of the Story engine shape the code below:

* Every box with a background colour (title band, callout, table header, zebra
  rows) is repainted as a stray strip on each *later* page of the same layout
  run. So the document is laid out as several runs ("parts") — a new part
  wherever a page break falls cleanly between blocks — and a long table
  continues in the next part under a repeated header row.
* It reports where a block *starts*, not where it ends. Each block therefore
  ends with an empty marker element whose own position says where it finished.
"""
import html
import re
import textwrap

from myapp.doc_blocks import (
    CALLOUTS, PALETTE as P, column_aligns, column_weights, date_text,
    document_title, host_of, inline_spans, parse_markdown, plain_text,
)
from myapp.file_convert import ConvertError

_PORTRAIT = (595, 842)        # A4, points
_LANDSCAPE = (842, 595)
_MARGINS = (54, 58, 54, 62)   # left, top, right, bottom
_MAX_PASSES = 8
_MIN_PAGES_FOR_CONTENTS = 3
_MIN_HEADINGS_FOR_CONTENTS = 4
_KEEP_WHOLE_LINES = 30        # a code block shorter than this is moved, not split


def _css():
    callouts = ''.join(
        f'div.callout.{kind} {{ background-color: #{bg}; border-left: 3.5pt solid #{stripe}; }}'
        f'div.callout.{kind} .cl {{ color: #{stripe}; }}'
        for kind, (stripe, bg, _label) in CALLOUTS.items()
    )
    return f"""
body {{ font-family: sans-serif; font-size: 10.5pt; color: #{P['body']}; line-height: 1.5; }}
p {{ margin: 0 0 7pt 0; }}
p.lead {{ font-size: 12pt; color: #{P['muted']}; line-height: 1.45; margin: 0 0 10pt 0; }}
h1, h2, h3, h4 {{ color: #{P['ink']}; margin: 0; line-height: 1.25; }}
h1.sec {{ font-size: 21pt; margin: 20pt 0 8pt 0; padding-bottom: 3pt; border-bottom: 2pt solid #{P['accent']}; }}
h2 {{ font-size: 15.5pt; color: #{P['accent_dark']}; margin: 17pt 0 6pt 0; padding-bottom: 2.5pt; border-bottom: 0.75pt solid #{P['mint']}; }}
h3 {{ font-size: 12.5pt; margin: 13pt 0 4pt 0; }}
h4 {{ font-size: 10.5pt; color: #{P['muted']}; margin: 10pt 0 3pt 0; }}
.plain {{ font-weight: normal; }}
.hero {{ background-color: #{P['deep']}; padding: 22pt 24pt 20pt 24pt; margin: 0 0 14pt 0; }}
.hero .kicker {{ font-size: 9pt; font-weight: bold; color: #{P['mint']}; letter-spacing: 1.5pt; margin: 0 0 7pt 0; }}
.hero h1 {{ font-size: 27pt; color: #ffffff; margin: 0; line-height: 1.2; }}
.hero .meta {{ font-size: 10pt; color: #{P['mint']}; margin: 9pt 0 0 0; }}
ul, ol {{ margin: 0 0 8pt 0; padding-left: 18pt; }}
li {{ margin: 0 0 3pt 0; }}
a {{ color: #{P['accent_dark']}; }}
.host {{ color: #{P['faint']}; font-size: 9pt; }}
code {{ font-family: monospace; font-size: 9.2pt; background-color: #{P['code_bg']}; color: #{P['accent_dark']}; }}
pre {{ font-family: monospace; font-size: 8.5pt; line-height: 1.4; color: #{P['ink']}; background-color: #{P['code_bg']}; border: 0.5pt solid #{P['rule']}; padding: 7pt 9pt; margin: 6pt 0 10pt 0; }}
blockquote {{ margin: 6pt 0 9pt 0; padding: 2pt 0 2pt 12pt; border-left: 2.5pt solid #{P['rule']}; color: #{P['muted']}; font-style: italic; }}
hr {{ border: 0; border-top: 0.75pt solid #{P['rule']}; margin: 12pt 0; }}
div.callout {{ margin: 8pt 0 11pt 0; padding: 8pt 12pt 6pt 12pt; }}
div.callout p {{ margin: 0 0 3pt 0; }}
.callout .cl {{ font-weight: bold; font-size: 8.5pt; letter-spacing: 1pt; }}
{callouts}
table.md {{ width: 100%; border-collapse: collapse; margin: 6pt 0 11pt 0; }}
table.md th {{ background-color: #{P['accent']}; color: #ffffff; font-weight: bold; font-size: 9.5pt; padding: 5pt 7pt; border: 0.5pt solid #{P['accent']}; }}
table.md td {{ font-size: 9.5pt; padding: 4.5pt 7pt; border-bottom: 0.5pt solid #{P['rule']}; }}
table.md tr.alt td {{ background-color: #{P['zebra']}; }}
table.md.dense th, table.md.dense td {{ font-size: 8.5pt; padding: 3.5pt 5pt; }}
p.chart {{ text-align: center; margin: 8pt 0 11pt 0; }}
.toc {{ margin: 4pt 0 14pt 0; padding: 10pt 16pt 6pt 16pt; background-color: #{P['soft']}; }}
.toc .toch {{ font-weight: bold; font-size: 11pt; color: #{P['accent_dark']}; margin: 0 0 5pt 0; }}
table.tocl {{ width: 100%; border-collapse: collapse; }}
table.tocl td {{ font-size: 10pt; padding: 2.5pt 0; border-bottom: 0.5pt solid #{P['mint']}; }}
table.tocl td.p {{ text-align: right; width: 36pt; color: #{P['muted']}; }}
table.tocl td.l2 {{ padding-left: 14pt; color: #{P['muted']}; }}
"""


# ───────────────────────────────────────────────────────────── html pieces ──

_LONG_TOKEN_RE = re.compile(r'(?<!\S)(?!https?://)\S{60,}')


def _breakable(text):
    """Give an unbroken string of 60+ characters (a hash, a base64 blob) places
    to wrap: the engine cannot split a "word", so it would run off the page and
    push a table column out of view. Web addresses wrap on their own and are
    left alone, so copying one still gives the real address."""
    return _LONG_TOKEN_RE.sub(
        lambda m: '​'.join(m.group(0)[i:i + 40] for i in range(0, len(m.group(0)), 40)), text,
    )


def spans_html(text):
    """Inline Markdown -> HTML. A link keeps its label and shows its domain, so
    a printed copy still says where the label points."""
    pieces = []
    for span in inline_spans(text):
        piece = html.escape(span.text if span.url else _breakable(span.text)).replace('\n', '<br/>')
        if span.code:
            piece = f'<code>{piece}</code>'
        if span.bold:
            piece = f'<b>{piece}</b>'
        if span.italic:
            piece = f'<i>{piece}</i>'
        if span.strike:
            piece = f'<s>{piece}</s>'
        if span.url:
            piece = f'<a href="{html.escape(span.url, quote=True)}">{piece}</a>'
            host = host_of(span.url)
            if host and host not in span.text and span.text.strip() != span.url:
                piece += f' <span class="host">({html.escape(host)})</span>'
        pieces.append(piece)
    return ''.join(pieces)


def _is_latin(text):
    """Non-Latin script comes from fallback fonts that only ship a regular
    weight; asking for bold there smears the glyphs, so those headings stay
    regular."""
    return all(ord(ch) < 0x250 or not ch.isalpha() for ch in text)


def _wrap_code(code):
    lines = []
    for line in code.expandtabs(4).split('\n'):
        lines.extend(textwrap.wrap(
            line, 88, replace_whitespace=False, drop_whitespace=False, break_long_words=True,
        ) or [''])
    return '\n'.join(lines)


def _marker(element_id):
    """An empty element whose position says where the block before it ended."""
    return f'<span id="{element_id}"></span>'


def _list_html(items, marker=''):
    """Nested <ul>/<ol> from flat item blocks (each has a depth and a kind)."""
    out = []
    stack = []   # (depth, tag)
    for number, item in enumerate(items):
        tag = 'ol' if item.ordered else 'ul'
        while stack and (stack[-1][0] > item.level or (stack[-1][0] == item.level and stack[-1][1] != tag)):
            out.append(f'</li></{stack.pop()[1]}>')
        if stack and stack[-1][0] == item.level:
            out.append('</li>')
        elif not stack or stack[-1][0] < item.level:
            start = f' start="{item.number}"' if tag == 'ol' and item.number > 1 else ''
            out.append(f'<{tag}{start}>')
            stack.append((item.level, tag))
        out.append(f'<li>{spans_html(item.text)}{marker if number == len(items) - 1 else ""}')
    while stack:
        out.append(f'</li></{stack.pop()[1]}>')
    return ''.join(out)


def _table_chunks(block, bid, cuts):
    """The table as chunks, split before every row number in ``cuts``. Each chunk
    repeats the header row, so a long table stays readable on every page it
    reaches. Returns [(html, last row marker id)]."""
    aligns = column_aligns(block)
    weights = column_weights(block)
    total = sum(weights) or 1
    widths = [f'{max(6, round(w / total * 100))}%' for w in weights]
    dense = ' dense' if len(block.header) >= 6 else ''

    def cell(tag, text, index, width='', extra=''):
        style = f'text-align:{aligns[index]};' + (f'width:{width};' if width else '')
        return f'<{tag} style="{style}">{spans_html(text)}{extra}</{tag}>'

    def head(chunk):
        cells = ''.join(cell('th', text, i, widths[i]) for i, text in enumerate(block.header))
        return f'<tr id="{bid}h{chunk}">{cells}</tr>'

    def row(number):
        css = ' class="alt"' if number % 2 else ''
        cells = block.rows[number]
        last = len(cells) - 1
        html_cells = ''.join(
            cell('td', text, i, extra=_marker(f'{bid}r{number}x') if i == last else '')
            for i, text in enumerate(cells)
        )
        return f'<tr id="{bid}r{number}"{css}>{html_cells}</tr>'

    bounds = [0] + sorted(c for c in set(cuts) if 0 < c < len(block.rows)) + [len(block.rows)]
    chunks = []
    for chunk in range(len(bounds) - 1):
        rows = ''.join(row(n) for n in range(bounds[chunk], bounds[chunk + 1]))
        last_marker = f'{bid}r{bounds[chunk + 1] - 1}x' if bounds[chunk + 1] > bounds[chunk] else f'{bid}h{chunk}'
        chunks.append((f'<table class="md{dense}">{head(chunk)}{rows}</table>', last_marker))
    return chunks


class _Build:
    """Turns blocks into HTML plus the assets (chart PNGs) it uses.

    The HTML comes back as a list of *parts*, each laid out from a fresh page
    (see the module notes). ``block_ids`` / ``block_ends`` hold, for every
    top-level block, the id of its first element and of its end marker.
    """

    def __init__(self, blocks, brand, title, created, branded=True):
        self.blocks = blocks
        self.brand = brand
        self.created = created
        self.title = title
        self.branded = branded
        self.charts = []
        self.headings = []     # (element id, level, plain text)
        self.block_ids = []
        self.block_ends = []
        self.tables = []       # (block id, number of rows)
        self.keep_whole = []   # ids of small blocks that should not be cut by a page break
        self.has_hero = False

    def html(self, toc_pages=None, breaks=frozenset(), plans=None):
        from myapp import doc_charts
        self.charts.clear()
        self.headings.clear()
        self.block_ids.clear()
        self.block_ends.clear()
        self.tables.clear()
        self.keep_whole.clear()

        parts = [[]]

        def emit(piece):
            parts[-1].append(piece)

        def split_now():
            if parts[-1]:
                parts.append([])

        blocks = list(self.blocks)
        start = 0
        while start < len(blocks) and blocks[start].kind == 'rule':
            start += 1
        hero_block = (
            blocks[start] if start < len(blocks)
            and blocks[start].kind == 'heading' and blocks[start].level == 1 else None
        )
        if self.branded and (hero_block is not None or self.title):
            self.has_hero = True
            text = plain_text(hero_block.text) if hero_block is not None else self.title
            plain = '' if _is_latin(text) else ' plain'
            emit(
                '<div class="hero">'
                + (f'<p class="kicker">{html.escape(self.brand.upper())} AI</p>' if self.brand else '') +
                f'<h1 class="{plain.strip()}" id="h0">{html.escape(text)}</h1>'
                f'<p class="meta">Prepared on {html.escape(date_text(self.created))}{_marker("herox")}</p></div>'
            )
            self.headings.append(('h0', 1, text))
            if hero_block is not None:
                blocks = blocks[start + 1:]
        if toc_pages is not None:
            emit(self._toc(toc_pages, blocks))

        first_text = True
        index = 0
        while index < len(blocks):
            block = blocks[index]
            bid = f'b{len(self.block_ids)}'
            first = f'h{len(self.headings)}' if block.kind == 'heading' else bid
            end = first if block.kind in ('heading', 'rule') else f'{bid}x'
            if first in breaks:
                split_now()
            self.block_ids.append(first)
            self.block_ends.append(end)

            if block.kind == 'item':
                run = []
                while index < len(blocks) and blocks[index].kind == 'item':
                    run.append(blocks[index])
                    index += 1
                emit(f'<div id="{bid}">{_list_html(run, _marker(end))}</div>')
                first_text = False
                continue
            index += 1
            if block.kind == 'heading':
                level = min(block.level, 4)
                self.headings.append((first, block.level, plain_text(block.text)))
                classes = []
                if block.level == 1:
                    classes.append('sec')
                if not _is_latin(block.text):
                    classes.append('plain')
                cls = f' class="{" ".join(classes)}"' if classes else ''
                emit(f'<h{level} id="{first}"{cls}>{spans_html(block.text)}</h{level}>')
                first_text = False
            elif block.kind == 'para':
                lead = ' class="lead"' if first_text and self.has_hero and len(block.text) <= 260 else ''
                emit(f'<p id="{bid}"{lead}>{spans_html(block.text)}{_marker(end)}</p>')
                first_text = False
            elif block.kind == 'table':
                for number, (table, last_marker) in enumerate(_table_chunks(block, bid, (plans or {}).get(bid, ()))):
                    if number > 0:
                        split_now()
                    emit(f'<div id="{bid}">{table}</div>' if number == 0 else f'<div>{table}</div>')
                    end = last_marker
                self.block_ends[-1] = end
                self.tables.append((bid, len(block.rows)))
                first_text = False
            elif block.kind == 'code':
                wrapped = _wrap_code(block.text)
                if wrapped.count('\n') < _KEEP_WHOLE_LINES:
                    self.keep_whole.append(bid)
                emit(f'<pre id="{bid}">{html.escape(wrapped)}{_marker(end)}</pre>')
                first_text = False
            elif block.kind == 'quote':
                if block.callout:
                    label = CALLOUTS[block.callout][2]
                    self.keep_whole.append(bid)
                    emit(
                        f'<div class="callout {block.callout}" id="{bid}">'
                        f'<p><span class="cl">{label.upper()}</span></p>'
                        f'<p>{spans_html(block.text)}{_marker(end)}</p></div>'
                    )
                else:
                    emit(f'<blockquote id="{bid}">{spans_html(block.text)}{_marker(end)}</blockquote>')
                first_text = False
            elif block.kind == 'rule':
                emit(f'<hr id="{bid}"/>')
            elif block.kind == 'chart':
                try:
                    png = doc_charts.chart_png(block.chart)
                except Exception:   # a chart that cannot be drawn must not sink the document
                    self.block_ids.pop()
                    self.block_ends.pop()
                    continue
                name = f'chart{len(self.charts)}.png'
                self.charts.append((name, png))
                emit(f'<p class="chart" id="{bid}"><img src="{name}" width="470"/>{_marker(end)}</p>')
                first_text = False
        return ['<html><body>' + '\n'.join(part) + '</body></html>' for part in parts if part]

    def _toc(self, toc_pages, blocks):
        rows = []
        # Entries come from the body headings in order; html() hands out their
        # ids the same way, so the page numbers line up.
        counter = 1 if self.has_hero else 0
        entries = []
        for block in blocks:
            if block.kind != 'heading':
                continue
            hid = f'h{counter}'
            counter += 1
            if block.level in (1, 2, 3):
                entries.append((hid, block))
        for number, (hid, block) in enumerate(entries):
            page = toc_pages.get(hid)
            klass = ' class="l2"' if block.level == 3 else ''
            extra = _marker('tocx') if number == len(entries) - 1 else ''
            rows.append(
                f'<tr><td{klass}>{html.escape(plain_text(block.text))}</td>'
                f'<td class="p">{page if page else ""}{extra}</td></tr>'
            )
        return (
            '<div class="toc" id="toc"><p class="toch">Contents</p>'
            '<table class="tocl">' + ''.join(rows) + '</table></div>'
        )


# ─────────────────────────────────────────────────────────────── rendering ──

def _layout_part(fitz, html_text, archive, size):
    width, height = size
    left, top, right, bottom = _MARGINS
    mediabox = fitz.Rect(0, 0, width, height)
    where = fitz.Rect(left, top, width - right, height - bottom)
    pages, boxes = {}, {}

    def rectfn(rect_num, filled):
        return mediabox, where, fitz.Identity

    def positionfn(position):
        if position.id and position.open_close & 1:
            pages.setdefault(position.id, position.page_num)
            boxes.setdefault(position.id, (position.rect[1], position.rect[3]))

    story = fitz.Story(html=html_text, user_css=_css(), archive=archive)
    return story.write_with_links(rectfn, positionfn), pages, boxes


def _layout(fitz, parts, archive, size):
    """Lay each part out from a fresh page and join them; returns
    (document, pages, boxes). ``pages`` maps every element id to the page it
    sits on, counted across the whole document; ``boxes`` to its (top, bottom)
    on that page, in points."""
    merged = None
    pages, boxes = {}, {}
    offset = 0
    for part in parts:
        document, part_pages, part_boxes = _layout_part(fitz, part, archive, size)
        for key, page in part_pages.items():
            pages[key] = page + offset
        boxes.update(part_boxes)
        offset += document.page_count
        if merged is None:
            merged = document
        else:
            merged.insert_pdf(document)
            document.close()
    return merged, pages, boxes


_TABLE_MARGIN = 6   # points above a table (table.md in the stylesheet)


def _plan_table(bid, row_count, pages, boxes, size):
    """Row numbers where a table should carry on in a new part, worked out from
    the measured height of every row: rows are packed onto each page until the
    next one would cross the bottom margin. A continuing page repeats the header
    row, so it holds one row fewer — which is exactly what accumulating breaks
    from successive layouts could not account for.

    Returns (rows to start a new page at, whether row 0 itself must move)."""
    head = boxes.get(f'{bid}h0')
    if not head or not row_count:
        return set(), False
    heights = []
    for number in range(row_count):
        box = boxes.get(f'{bid}r{number}')
        whole = box and pages.get(f'{bid}r{number}') == pages.get(f'{bid}r{number}x')
        heights.append(box[1] - box[0] if whole and box[1] > box[0] else None)
    known = sorted(h for h in heights if h)
    if not known:
        return set(), False
    typical = known[len(known) // 2]
    head_height = head[1] - head[0]
    top, bottom = _MARGINS[1], size[1] - _MARGINS[3] - 2
    fresh = top + _TABLE_MARGIN + head_height       # first row's top on a new page
    y = head[1]
    cuts, moves = set(), False
    for number, height in enumerate(heights):
        height = height or typical
        if y + height > bottom:
            if number == 0:
                moves = True
            else:
                cuts.add(number)
            y = fresh + height
        else:
            y += height
    return cuts, moves


def _new_breaks(build, pages, boxes, size, already, remembered):
    """What the next layout pass should change. Returns (breaks, plans):

    * ``breaks`` — element ids that should start a new part (a fresh page):
      a heading whose text starts on the next page moves there with it, as does
      a short callout or code block cut by the page edge, a table whose header
      is stranded away from its first row, and any block that begins a page with
      nothing spilling onto it (every clean page break becomes a part boundary,
      which keeps the engine from repainting backgrounds onto later pages);
    * ``plans`` — for each table, the rows to continue from on a new page.
      ``remembered`` holds the rows a layout once cut in half, so a row that
      needed moving stays moved.
    """
    found = set()
    plans = {}
    ids = build.block_ids

    def lead_in(bid):
        """What moves ``bid`` to a new page: itself, unless a heading sits right
        above it, in which case the heading goes too — a lone heading left
        behind is exactly what the first rule removes."""
        position = ids.index(bid) if bid in ids else -1
        before = ids[position - 1] if position > 0 else ''
        if before.startswith('h') and before != 'h0':
            return before
        return bid

    for number, hid in enumerate(ids):
        if hid.startswith('h') and hid != 'h0' and hid not in already and number + 1 < len(ids):
            here, after = pages.get(hid), pages.get(ids[number + 1])
            if here and after and after > here:
                found.add(hid)

    for bid, row_count in build.tables:
        head, first = pages.get(f'{bid}h0'), pages.get(f'{bid}r0')
        if head and first and first > head:
            found.add(lead_in(bid))
        cuts, moves = _plan_table(bid, row_count, pages, boxes, size)
        if moves:
            found.add(lead_in(bid))
        for number in range(row_count):
            start, end = pages.get(f'{bid}r{number}'), pages.get(f'{bid}r{number}x')
            if start is not None and end is not None and end != start:
                if number == 0:
                    found.add(lead_in(bid))
                else:
                    remembered.setdefault(bid, set()).add(number)
        plans[bid] = frozenset(cuts | remembered.get(bid, set()))

    for bid in build.keep_whole:
        start, end = pages.get(bid), pages.get(f'{bid}x')
        if start and end and end != start:
            found.add(lead_in(bid))

    previous_end = pages.get('tocx') or (pages.get('herox') if build.has_hero else None)
    previous_first = ''
    for first_id, end_id in zip(build.block_ids, build.block_ends):
        start, end = pages.get(first_id), pages.get(end_id)
        if start is None:
            continue
        orphan_heading = previous_first.startswith('h') and previous_first in found
        if previous_end is not None and start > previous_end and not orphan_heading:
            found.add(first_id)
        previous_end = end if end is not None else start
        previous_first = first_id
    return found - already, plans


def _footer_html(label, number, count):
    muted = '#' + P['muted']
    return (
        f'<table style="width:100%;border-collapse:collapse;font-family:sans-serif;font-size:8pt;color:{muted}">'
        f'<tr><td>{html.escape(label)}</td>'
        f'<td style="text-align:right">Page {number} of {count}</td></tr></table>'
    )


def _clip(text, limit):
    text = ' '.join((text or '').split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _outline(build, pages):
    entries = []
    previous = 0
    levels = [level for hid, level, _ in build.headings if hid in pages]
    if not levels:
        return entries
    base = min(levels)
    for hid, level, text in build.headings:
        if hid not in pages:
            continue
        depth = min(level - base + 1, previous + 1)
        entries.append([depth, _clip(text, 80), pages[hid]])
        previous = depth
    return entries


def render_pdf(content, *, title='', brand='', created=None, blocks=None,
               landscape=False, branded=True):
    """Markdown-ish text -> PDF bytes. ``branded=False`` drops the title band,
    the date and the brand name (for converting someone's own file)."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise ConvertError('PDF generation is temporarily unavailable.') from exc

    blocks = parse_markdown(content) if blocks is None else blocks
    size = _LANDSCAPE if landscape else _PORTRAIT
    if not blocks and not title:
        # Nothing to say: an empty page with no footer, so the file really is
        # blank (a scan check on it must find no text at all).
        blank = fitz.open()
        blank.new_page(width=size[0], height=size[1])
        data = blank.tobytes()
        blank.close()
        return data

    page_title = title or document_title(blocks)
    build = _Build(blocks, brand, title if not document_title(blocks) else '', created, branded)
    heading_blocks = [b for b in blocks if b.kind == 'heading']

    # Layout runs until it settles: a contents list shifts the pages it points
    # at, and every part boundary added below shifts them again, so each pass
    # feeds the next (a handful at most).
    breaks = set()
    plans = {}
    remembered = {}
    toc_pages = None
    doc = pages = None
    decided = False
    for _ in range(_MAX_PASSES):
        if doc is not None:
            doc.close()
        parts = build.html(toc_pages, frozenset(breaks), plans)
        archive = fitz.Archive()
        for name, png in build.charts:
            archive.add(png, name)
        doc, pages, boxes = _layout(fitz, parts, archive, size)

        if not decided:
            decided = True
            body_headings = [b for b in heading_blocks if b.level in (2, 3)]
            if doc.page_count >= _MIN_PAGES_FOR_CONTENTS and len(body_headings) >= _MIN_HEADINGS_FOR_CONTENTS:
                toc_pages = {}
                continue

        new_breaks, new_plans = _new_breaks(build, pages, boxes, size, breaks, remembered)
        new_pages = {hid: pages[hid] for hid, _, _ in build.headings if hid in pages}
        if not new_breaks and new_plans == plans and (toc_pages is None or new_pages == toc_pages):
            break
        breaks |= new_breaks
        plans = new_plans
        if toc_pages is not None:
            toc_pages = new_pages

    count = doc.page_count
    left, top, right, bottom = _MARGINS
    width, height = size
    if branded and brand:
        label = f'{_clip(page_title, 60)} · {brand} AI' if page_title else f'{brand} AI'
    else:
        label = _clip(page_title, 80)
    rule_color = tuple(int(P['rule'][i:i + 2], 16) / 255 for i in (0, 2, 4))
    for number, page in enumerate(doc, 1):
        rect = fitz.Rect(left, height - bottom + 16, width - right, height - 20)
        page.draw_line((rect.x0, rect.y0 - 7), (rect.x1, rect.y0 - 7), color=rule_color, width=0.5)
        page.insert_htmlbox(rect, _footer_html(label, number, count), css='body{margin:0}')

    outline = _outline(build, pages)
    if outline:
        try:
            doc.set_toc(outline)
        except (ValueError, RuntimeError):
            pass
    metadata = {'producer': f'{brand} AI', 'creator': f'{brand} AI'} if brand else {}
    if branded and brand:
        metadata['author'] = brand
    if page_title:
        metadata['title'] = page_title
    doc.set_metadata(metadata)
    try:   # keep only the glyphs used: the fallback fonts are large
        doc.subset_fonts()
    except Exception:
        pass
    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return data
