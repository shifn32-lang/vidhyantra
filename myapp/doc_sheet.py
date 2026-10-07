"""Designed Excel workbooks from table rows.

A styled header row (kept in row 1, frozen, with filter buttons), banded rows,
a bolded totals row when the data has one, column widths that fit, wrapped long
text, print setup (fit to one page wide, header repeated on every page), and
cells stored as *real* numbers, percentages and dates — formatted so they look
exactly like the text the data came with, but work in formulas and sorting.
"""
import io
import re
import unicodedata
from datetime import date, datetime, timezone

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

from myapp.doc_blocks import PALETTE as P, clean_text, stamp

MAX_COLUMN_WIDTH = 60
MIN_COLUMN_WIDTH = 8
_CURRENCIES = {'₹': '₹', 'rs': '₹', 'inr': '₹', '$': '$', 'usd': '$', '€': '€', '£': '£', '¥': '¥'}
# Thousands separators must be genuine: 1,234,567 (western) or 12,34,567
# (Indian). '12,34' is more likely a decimal comma, so it stays text.
_NUMBER_RE = re.compile(
    r'^(?P<sign>[-+−–]?)\s*(?P<cur>₹|Rs\.?|INR|USD|\$|€|£|¥)?\s*(?P<sign2>[-+−–]?)\s*'
    r'(?P<int>\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+)(?P<dec>\.\d+)?\s*(?P<pct>%)?$',
    re.I,
)
_PAREN_RE = re.compile(r'^\((?P<inner>[^()]+)\)$')
_ISO_DATE_RE = re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
# Digit strings this long are identifiers (phone, account, order numbers), not
# quantities: they stay text so no digit is lost and no leading zero dropped.
_ID_LENGTH = 10


def _typed(value):
    """A cell the way a spreadsheet user would have typed it: ('₹7,500' ->
    (7500, '"₹"#,##0')). The number format reproduces the original look, so
    nothing changes on screen, but the cell now works in formulas and sorts
    properly. Anything not clearly a number or a date stays text."""
    text = clean_text(str(value if value is not None else '')).strip()
    if not text:
        return '', None
    if text.startswith('='):
        return text, None        # a formula the model wrote: keep it live

    iso = _ISO_DATE_RE.match(text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))), 'yyyy-mm-dd'
        except ValueError:
            return text, None

    parens = _PAREN_RE.match(text)
    match = _NUMBER_RE.match(parens.group('inner').strip() if parens else text)
    if not match:
        return text, None
    whole, decimals = match.group('int'), match.group('dec') or ''
    digits = whole.replace(',', '')
    currency = _CURRENCIES.get((match.group('cur') or '').lower().rstrip('.'), '')
    if not decimals and not currency and ',' not in whole and len(digits) >= _ID_LENGTH:
        return text, None
    if not decimals and len(digits) > 1 and digits.startswith('0'):
        return text, None        # '007', '0123': leading zeros mean text
    places = max(0, len(decimals) - 1)
    number = float(digits + decimals)
    if parens or any(ch in '-−–' for ch in match.group('sign') + match.group('sign2')):
        number = -number
    if match.group('pct'):
        # Rounded so 8.5% is stored as 0.085, not 0.08500000000000001.
        return round(number / 100, 12), ('0.' + '0' * places + '%') if places else '0%'
    grouped = ',' in whole
    pattern = '#,##0' if (grouped or currency) else None
    if places:
        pattern = (pattern or '0') + '.' + '0' * places
    if currency:
        pattern = f'"{currency}"' + (pattern or '0')
    return (number if places else int(number)), pattern


def _display_width(text):
    total = 0
    for ch in str(text):
        if unicodedata.category(ch) in ('Mn', 'Me', 'Cf'):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return total


def _safe_title(title):
    cleaned = re.sub(r'[\[\]:*?/\\]', ' ', title or '').strip()
    return (cleaned or 'Data')[:31]


def render_xlsx(rows, *, title='', brand='', created=None, branded=True):
    """``rows`` (first row = headers) -> a styled .xlsx as bytes."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = _safe_title(title)
    sheet.sheet_view.showGridLines = False

    header_fill = PatternFill('solid', fgColor=P['accent'])
    band_fill = PatternFill('solid', fgColor=P['zebra'])
    total_fill = PatternFill('solid', fgColor=P['soft'])
    thin = Side(style='thin', color=P['rule'])
    columns = max((len(row) for row in rows), default=0)

    widths = [MIN_COLUMN_WIDTH] * columns
    for r_index, row in enumerate(rows, start=1):
        header = r_index == 1
        label = str(row[0]).strip().lower() if row and row[0] is not None else ''
        total = not header and label.startswith(('total', 'grand total', 'sum', 'average'))
        for c_index in range(columns):
            raw = row[c_index] if c_index < len(row) else ''
            if header:
                value, number_format = clean_text(str(raw or '')).strip(), None
            else:
                value, number_format = _typed(raw)
            cell = sheet.cell(row=r_index, column=c_index + 1, value=value if value != '' else None)
            if number_format:
                cell.number_format = number_format
            text = str(value)
            widths[c_index] = max(widths[c_index], min(_display_width(text) + 3, MAX_COLUMN_WIDTH))
            numeric = isinstance(value, (int, float, date, datetime)) and not isinstance(value, bool)
            if header:
                cell.font = Font(bold=True, color=P['white'], size=11)
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                cell.border = Border(bottom=Side(style='medium', color=P['accent_dark']))
            else:
                cell.font = Font(bold=total, color=P['body'], size=11)
                cell.alignment = Alignment(
                    horizontal='right' if numeric else 'left', vertical='top',
                    wrap_text=_display_width(text) > MAX_COLUMN_WIDTH - 3,
                )
                cell.border = Border(bottom=thin, top=Side(style='thin', color=P['accent']) if total else None)
                if total:
                    cell.fill = total_fill
                elif r_index % 2 == 1:
                    cell.fill = band_fill
    if rows:
        sheet.row_dimensions[1].height = 26
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    if rows and columns:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = f'A1:{get_column_letter(columns)}{len(rows)}'
        sheet.print_title_rows = '1:1'
        sheet.page_setup.orientation = 'landscape' if columns > 6 else 'portrait'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        if branded and brand:
            sheet.oddFooter.left.text = f'{brand} AI'
            sheet.oddFooter.left.size = 8
        sheet.oddFooter.right.text = 'Page &P of &N'
        sheet.oddFooter.right.size = 8

    properties = workbook.properties
    properties.title = title or sheet.title
    properties.creator = brand if branded else ''
    properties.lastModifiedBy = brand if branded else ''
    properties.description = f'Created with {brand} AI' if brand else ''
    properties.created = properties.modified = stamp(created).astimezone(timezone.utc).replace(tzinfo=None)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
