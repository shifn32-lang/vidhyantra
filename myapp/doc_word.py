"""Designed Word documents from Markdown-ish text.

Real Word structure, not just formatted text: headings use the built-in
Heading styles (so the navigation pane and a table of contents work), lists are
real lists with numbering that restarts for each list, tables repeat their
header row across pages, links are clickable, and the footer carries live page
numbers. On top of that: a title block, shaded callouts and code blocks,
zebra-striped tables and embedded charts.

python-docx covers the basics; the rest is small pieces of OOXML. Word is strict
about the order of properties inside an element, so every hand-built element
goes through ``_put``, which inserts it at its place in the schema order.
"""
import io
from datetime import timezone

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from myapp.doc_blocks import (
    CALLOUTS, PALETTE as P, column_aligns, column_weights, date_text,
    host_of, inline_spans, parse_markdown, plain_text, stamp,
)

BODY_FONT = 'Calibri'
CODE_FONT = 'Consolas'
INDIC_FONT = 'Nirmala UI'     # Word's own Indic-script font on Windows
TEXT_WIDTH_CM = 16.6          # A4 (21 cm) less 2 x 2.2 cm margins

# Child order inside the elements below (ECMA-376). Word rejects some files
# whose properties are out of order, so nothing is appended blindly.
_PPR = (
    'pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr',
    'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap',
    'overflowPunct', 'topLinePunct', 'autoSpaceDE', 'autoSpaceDN', 'bidi', 'adjustRightInd',
    'snapToGrid', 'spacing', 'ind', 'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc',
    'textDirection', 'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr',
    'sectPr', 'pPrChange',
)
_RPR = (
    'rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike', 'dstrike', 'outline',
    'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid', 'vanish', 'webHidden', 'color', 'spacing',
    'w', 'kern', 'position', 'sz', 'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText',
    'vertAlign', 'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath',
)
_TCPR = (
    'cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd', 'noWrap', 'tcMar',
    'textDirection', 'tcFitText', 'vAlign', 'hideMark',
)
_TBLPR = (
    'tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize',
    'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar',
    'tblLook', 'tblCaption', 'tblDescription',
)
_TRPR = (
    'cnfStyle', 'divId', 'gridBefore', 'gridAfter', 'wBefore', 'wAfter', 'cantSplit', 'trHeight',
    'tblHeader', 'tblCellSpacing', 'jc', 'hidden',
)


def _put(parent, child, order):
    """Insert ``child`` into ``parent`` at its place in ``order``, replacing any
    element of the same kind already there."""
    local = child.tag.split('}')[1]
    for existing in parent.findall(child.tag):
        parent.remove(existing)
    later = {qn(f'w:{name}') for name in order[order.index(local) + 1:]}
    for index, existing in enumerate(parent):
        if existing.tag in later:
            parent.insert(index, child)
            return child
    parent.append(child)
    return child


def _el(tag, **attrs):
    element = OxmlElement(tag)
    for key, value in attrs.items():
        element.set(qn(f'w:{key}'), str(value))
    return element


def _rgb(hex6):
    return RGBColor.from_string(hex6.upper())


# ───────────────────────────────────────────────────────── low-level pieces ──

def _fonts(rpr, name, complex_script=INDIC_FONT):
    """Pin a run/style to ``name`` (dropping the theme font that would override
    it) and give complex scripts a font that has the glyphs."""
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = _put(rpr, _el('w:rFonts'), _RPR)
    for attr in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'):
        rfonts.attrib.pop(qn(f'w:{attr}'), None)
    for attr, value in (('ascii', name), ('hAnsi', name), ('eastAsia', name), ('cs', complex_script)):
        rfonts.set(qn(f'w:{attr}'), value)


def _shade_paragraph(paragraph, fill):
    _put(paragraph._p.get_or_add_pPr(), _el('w:shd', val='clear', color='auto', fill=fill), _PPR)


def _border_paragraph(paragraph, sides, color, size, space=1):
    """Borders on ``sides`` ('top', 'left', 'bottom', 'right'); ``size`` is in
    eighths of a point."""
    borders = _el('w:pBdr')
    for side in ('top', 'left', 'bottom', 'right'):
        if side in sides:
            borders.append(_el(f'w:{side}', val='single', sz=size, space=space, color=color))
    _put(paragraph._p.get_or_add_pPr(), borders, _PPR)


def _shade_run(run, fill):
    _put(run._r.get_or_add_rPr(), _el('w:shd', val='clear', color='auto', fill=fill), _RPR)


def _letter_spacing(run, points):
    _put(run._r.get_or_add_rPr(), _el('w:spacing', val=int(points * 20)), _RPR)


def _cell_fill(cell, fill):
    _put(cell._tc.get_or_add_tcPr(), _el('w:shd', val='clear', color='auto', fill=fill), _TCPR)


def _field(paragraph, instruction, size=None, color=None):
    """A live field (PAGE, NUMPAGES …) as the three runs Word expects."""
    def piece(*children):
        run = paragraph.add_run()
        if size:
            run.font.size = size
        if color:
            run.font.color.rgb = color
        for child in children:
            run._r.append(child)

    instr = _el('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = f' {instruction} '
    placeholder = _el('w:t')
    placeholder.text = '1'
    piece(_el('w:fldChar', fldCharType='begin'), instr)
    piece(_el('w:fldChar', fldCharType='separate'), placeholder)
    piece(_el('w:fldChar', fldCharType='end'))


def _hyperlink(paragraph, url, text, *, size=None, bold=False, italic=False):
    relationship = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = _el('w:hyperlink')
    link.set(qn('r:id'), relationship)
    link.set(qn('w:history'), '1')
    run = OxmlElement('w:r')
    rpr = OxmlElement('w:rPr')
    run.append(rpr)
    if bold:
        _put(rpr, _el('w:b'), _RPR)
    if italic:
        _put(rpr, _el('w:i'), _RPR)
    _put(rpr, _el('w:color', val=P['accent_dark']), _RPR)
    if size:
        _put(rpr, _el('w:sz', val=int(size.pt * 2)), _RPR)
        _put(rpr, _el('w:szCs', val=int(size.pt * 2)), _RPR)
    _put(rpr, _el('w:u', val='single'), _RPR)
    piece = _el('w:t')
    piece.set(qn('xml:space'), 'preserve')
    piece.text = text.replace('\n', ' ')
    run.append(piece)
    link.append(run)
    paragraph._p.append(link)


def _add_spans(paragraph, text, *, size=None, color=None, bold=False, italic=False):
    """Inline Markdown as runs: bold, italic, code, strike and real links. A
    link also shows its domain, so a printed copy says where it points."""
    for span in inline_spans(text):
        if span.url:
            _hyperlink(paragraph, span.url, span.text, size=size, bold=bold or span.bold,
                       italic=italic or span.italic)
            host = host_of(span.url)
            if host and host not in span.text and span.text.strip() != span.url:
                note = paragraph.add_run(f' ({host})')
                note.font.size = Pt(max(7, (size.pt if size else 11) - 2))
                note.font.color.rgb = _rgb(P['faint'])
            continue
        run = paragraph.add_run(span.text)
        if span.bold or bold:
            run.bold = True
        if span.italic or italic:
            run.italic = True
        if span.strike:
            run.font.strike = True
        if color:
            run.font.color.rgb = color
        if size:
            run.font.size = size
        if span.code:
            _fonts(run._r.get_or_add_rPr(), CODE_FONT, CODE_FONT)
            run.font.size = Pt(max(7, (size.pt if size else 11) - 1))
            run.font.color.rgb = _rgb(P['accent_dark'])
            _shade_run(run, P['code_bg'])


def _spacing(paragraph, before=None, after=None, line=None):
    fmt = paragraph.paragraph_format
    if before is not None:
        fmt.space_before = Pt(before)
    if after is not None:
        fmt.space_after = Pt(after)
    if line is not None:
        fmt.line_spacing = line


# ──────────────────────────────────────────────────────────── set-up pieces ──

def _style_font(style, *, size=None, bold=None, color=None, name=BODY_FONT):
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = _rgb(color)
    _fonts(style.element.get_or_add_rPr(), name)


def _setup_styles(document):
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin = section.bottom_margin = Cm(2.2)

    styles = document.styles
    normal = styles['Normal']
    _style_font(normal, size=11, color=P['body'])
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for name, size, color, before, after in (
        ('Heading 1', 20, P['ink'], 20, 8),
        ('Heading 2', 15, P['accent_dark'], 16, 6),
        ('Heading 3', 12.5, P['ink'], 12, 4),
        ('Heading 4', 11, P['muted'], 10, 3),
    ):
        style = styles[name]
        _style_font(style, size=size, bold=True, color=color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.line_spacing = 1.1

    for name in ('List Bullet', 'List Bullet 2', 'List Bullet 3',
                 'List Number', 'List Number 2', 'List Number 3'):
        styles[name].paragraph_format.space_after = Pt(3)


def _setup_footer(document, label):
    footer = document.sections[0].footer
    paragraph = footer.paragraphs[0]
    paragraph.text = ''
    # Normal, not the built-in Footer style: that one brings centre and right
    # tab stops of its own, which would catch the tab before ours does.
    paragraph.style = document.styles['Normal']
    paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
    _border_paragraph(paragraph, ('top',), P['rule'], 4, space=6)
    muted, small = _rgb(P['muted']), Pt(8.5)
    run = paragraph.add_run(label + '\tPage ')
    run.font.size, run.font.color.rgb = small, muted
    _field(paragraph, 'PAGE', small, muted)
    run = paragraph.add_run(' of ')
    run.font.size, run.font.color.rgb = small, muted
    _field(paragraph, 'NUMPAGES', small, muted)


def _set_properties(document, title, brand, created, branded=True):
    moment = stamp(created).astimezone(timezone.utc).replace(tzinfo=None)
    props = document.core_properties
    props.title = title or ''
    props.author = brand if branded else ''
    props.last_modified_by = brand if branded else ''
    props.comments = f'Created with {brand} AI' if brand else ''
    props.created = props.modified = moment


# ───────────────────────────────────────────────────────────── block writers ──

class _Lists:
    """Numbered lists that restart: Word's List Number style shares one counter
    across the whole document, so each new list gets its own numbering instance."""

    def __init__(self, document):
        self.document = document
        self.numbering = document.part.numbering_part.numbering_definitions._numbering
        self.active = {}

    def reset(self):
        self.active.clear()

    def _new_num(self, style_name, start):
        base = self.document.styles[style_name].element.pPr.numPr.numId.val
        abstract = self.numbering.num_having_numId(base).abstractNumId.val
        num = self.numbering.add_num(abstract)
        num.add_lvlOverride(ilvl=0).add_startOverride(start)
        return num.numId

    def apply(self, paragraph, item, style_name):
        if not item.ordered:
            return
        if item.number <= 1 or item.level not in self.active:
            self.active[item.level] = self._new_num(style_name, max(1, item.number))
        numpr = paragraph._p.get_or_add_pPr().get_or_add_numPr()
        numpr.get_or_add_ilvl().val = 0
        numpr.get_or_add_numId().val = self.active[item.level]


def _title_block(document, text, brand, created):
    if brand:
        kicker = document.add_paragraph()
        run = kicker.add_run(f'{brand.upper()} AI')
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = _rgb(P['accent'])
        _letter_spacing(run, 1.5)
        _spacing(kicker, before=0, after=2)
        kicker.paragraph_format.keep_with_next = True

    # Still a real Heading 1, so the navigation pane lists the document title.
    heading = document.add_paragraph(style='Heading 1')
    run = heading.add_run(text.replace('\n', ' '))
    run.font.size = Pt(28)
    _spacing(heading, before=0, after=4)

    meta = document.add_paragraph()
    run = meta.add_run(f'Prepared on {date_text(created)}')
    run.font.size = Pt(10)
    run.font.color.rgb = _rgb(P['muted'])
    _border_paragraph(meta, ('bottom',), P['accent'], 12, space=6)
    _spacing(meta, before=0, after=14)


def _heading(document, block):
    level = min(block.level, 4)
    paragraph = document.add_paragraph(style=f'Heading {level}')
    _add_spans(paragraph, block.text.replace('\n', ' '))
    if level == 1:
        _border_paragraph(paragraph, ('bottom',), P['accent'], 12, space=3)
    elif level == 2:
        _border_paragraph(paragraph, ('bottom',), P['mint'], 6, space=2)


def _item(document, lists, block):
    depth = min(block.level, 2)
    base = 'List Number' if block.ordered else 'List Bullet'
    style = base if depth == 0 else f'{base} {depth + 1}'
    paragraph = document.add_paragraph(style=style)
    _add_spans(paragraph, block.text)
    lists.apply(paragraph, block, style)


def _callout(document, block):
    stripe, background, label = CALLOUTS[block.callout]
    paragraph = document.add_paragraph()
    run = paragraph.add_run(label.upper())
    run.bold = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = _rgb(stripe)
    _letter_spacing(run, 1)
    paragraph.add_run().add_break()
    _add_spans(paragraph, block.text)
    _shade_paragraph(paragraph, background)
    _border_paragraph(paragraph, ('left',), stripe, 24, space=8)
    paragraph.paragraph_format.left_indent = Cm(0.35)
    paragraph.paragraph_format.right_indent = Cm(0.1)
    paragraph.paragraph_format.keep_together = True
    _spacing(paragraph, before=6, after=10)


def _quote(document, block):
    paragraph = document.add_paragraph()
    _add_spans(paragraph, block.text, italic=True, color=_rgb(P['muted']))
    _border_paragraph(paragraph, ('left',), P['rule'], 18, space=8)
    paragraph.paragraph_format.left_indent = Cm(0.5)
    _spacing(paragraph, before=4, after=8)


def _code(document, block):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(block.text.expandtabs(4))
    _fonts(run._r.get_or_add_rPr(), CODE_FONT, CODE_FONT)
    run.font.size = Pt(9)
    run.font.color.rgb = _rgb(P['ink'])
    _shade_paragraph(paragraph, P['code_bg'])
    _border_paragraph(paragraph, ('top', 'left', 'bottom', 'right'), P['rule'], 4, space=5)
    paragraph.paragraph_format.left_indent = Cm(0.2)
    paragraph.paragraph_format.right_indent = Cm(0.2)
    paragraph.paragraph_format.line_spacing = 1.0
    if block.text.count('\n') < 40:
        paragraph.paragraph_format.keep_together = True
    _spacing(paragraph, before=4, after=10)


def _rule(document):
    paragraph = document.add_paragraph()
    _border_paragraph(paragraph, ('bottom',), P['rule'], 6, space=1)
    _spacing(paragraph, before=2, after=8, line=0.6)


def _table(document, block):
    from docx.enum.text import WD_ALIGN_PARAGRAPH as ALIGN
    columns = len(block.header)
    aligns = column_aligns(block)
    weights = column_weights(block)
    widths = [TEXT_WIDTH_CM * weight / (sum(weights) or 1) for weight in weights]
    size = Pt(9 if columns >= 7 else 10)
    alignment = {'left': ALIGN.LEFT, 'right': ALIGN.RIGHT, 'center': ALIGN.CENTER}

    table = document.add_table(rows=1 + len(block.rows), cols=columns)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    properties = table._tbl.tblPr
    total_twips = int(TEXT_WIDTH_CM / 2.54 * 1440)
    _put(properties, _el('w:tblW', w=total_twips, type='dxa'), _TBLPR)
    borders = _el('w:tblBorders')
    borders.append(_el('w:top', val='single', sz=8, space=0, color=P['accent']))
    borders.append(_el('w:left', val='nil'))
    borders.append(_el('w:bottom', val='single', sz=6, space=0, color=P['rule']))
    borders.append(_el('w:right', val='nil'))
    borders.append(_el('w:insideH', val='single', sz=4, space=0, color=P['rule']))
    borders.append(_el('w:insideV', val='nil'))
    _put(properties, borders, _TBLPR)
    margins = _el('w:tblCellMar')
    for side, width in (('top', 70), ('left', 110), ('bottom', 70), ('right', 110)):
        margins.append(_el(f'w:{side}', w=width, type='dxa'))
    _put(properties, margins, _TBLPR)

    for index, width in enumerate(widths):
        table.columns[index].width = Cm(width)   # the grid Word uses for a fixed layout

    def fill_row(row, cells, header=False, shade=None, bold=False):
        trpr = row._tr.get_or_add_trPr()
        _put(trpr, _el('w:cantSplit'), _TRPR)
        if header:
            _put(trpr, _el('w:tblHeader'), _TRPR)
        for index, text in enumerate(cells):
            cell = row.cells[index]
            cell.width = Cm(widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = alignment[aligns[index]]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.1
            _add_spans(paragraph, text, size=size, bold=header or bold,
                       color=_rgb(P['white']) if header else None)
            if header:
                _cell_fill(cell, P['accent'])
            elif shade:
                _cell_fill(cell, shade)

    fill_row(table.rows[0], block.header, header=True)
    for number, cells in enumerate(block.rows):
        total = bool(cells) and plain_text(cells[0]).strip().lower().startswith(('total', 'grand total'))
        fill_row(
            table.rows[number + 1], cells, bold=total,
            shade=P['soft'] if total else (P['zebra'] if number % 2 else None),
        )
    spacer = document.add_paragraph()
    _spacing(spacer, before=0, after=6, line=0.5)


def _chart(document, block):
    from myapp import doc_charts
    try:
        png = doc_charts.chart_png(block.chart)
    except Exception:   # a chart that cannot be drawn must not sink the document
        return
    document.add_picture(io.BytesIO(png), width=Cm(TEXT_WIDTH_CM - 0.4))
    paragraph = document.paragraphs[-1]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(paragraph, before=4, after=10)
    picture = paragraph._p.xpath('.//wp:docPr')
    if picture:
        picture[0].set('descr', block.chart.get('title') or 'Chart')


# ─────────────────────────────────────────────────────────────── rendering ──

def render_docx(content, *, title='', brand='', created=None, blocks=None, branded=True):
    """Markdown-ish text -> a .docx as bytes. ``branded=False`` drops the title
    block, the date and the brand name (for converting someone's own file)."""
    blocks = parse_markdown(content) if blocks is None else blocks
    document = Document()
    _setup_styles(document)

    start = 0
    while start < len(blocks) and blocks[start].kind == 'rule':
        start += 1
    lead = blocks[start] if start < len(blocks) else None
    has_title_block = branded and lead is not None and lead.kind == 'heading' and lead.level == 1
    page_title = (plain_text(lead.text) if has_title_block else '') or title
    if not branded and lead is not None and lead.kind == 'heading' and lead.level == 1:
        page_title = page_title or plain_text(lead.text)
    if has_title_block:
        blocks = blocks[start + 1:]
    if has_title_block or (branded and title):
        _title_block(document, page_title, brand, created)

    _set_properties(document, page_title, brand, created, branded)
    if branded and brand:
        footer_label = f'{page_title[:60]} · {brand} AI' if page_title else f'{brand} AI'
    else:
        footer_label = page_title[:80]
    _setup_footer(document, footer_label)

    lists = _Lists(document)
    first_text = True
    for block in blocks:
        if block.kind != 'item':
            lists.reset()
        if block.kind == 'heading':
            _heading(document, block)
            first_text = False
        elif block.kind == 'para':
            paragraph = document.add_paragraph()
            if first_text and (has_title_block or (branded and title)) and len(block.text) <= 260:
                _add_spans(paragraph, block.text, size=Pt(12.5), color=_rgb(P['muted']))
                _spacing(paragraph, after=10)
            else:
                _add_spans(paragraph, block.text)
            first_text = False
        elif block.kind == 'item':
            _item(document, lists, block)
            first_text = False
        elif block.kind == 'table':
            _table(document, block)
            first_text = False
        elif block.kind == 'quote':
            (_callout if block.callout else _quote)(document, block)
            first_text = False
        elif block.kind == 'code':
            _code(document, block)
            first_text = False
        elif block.kind == 'rule':
            _rule(document)
        elif block.kind == 'chart':
            _chart(document, block)
            first_text = False

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
