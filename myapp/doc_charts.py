"""Charts for generated documents, drawn without any plotting library.

A ```chart block (see doc_blocks.parse_chart) becomes an SVG that PyMuPDF — which
is already a dependency — rasterises to a PNG. The PDF and Word writers embed
that picture; the PowerPoint writer builds a native, editable chart instead
(doc_render.py), so this module is only the picture path.

Every value comes straight from the spec; nothing is estimated, smoothed or
filled in, and a missing value is a visible gap, never a zero.
"""
import math
from xml.sax.saxutils import escape

from myapp.doc_blocks import CHART_COLORS, PALETTE, format_value

WIDTH, HEIGHT = 760, 400
_FONT = 'Helvetica, Arial, sans-serif'


def _hex(name):
    return '#' + PALETTE[name]


def _color(index):
    return '#' + CHART_COLORS[index % len(CHART_COLORS)]


def _text_width(text, size):
    return len(text) * size * 0.56


def _clip(text, limit):
    text = (text or '').strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _nice_ticks(low, high, target=5):
    if high <= low:
        high = low + 1
    span = high - low
    step = 10 ** math.floor(math.log10(span / target))
    for factor in (1, 2, 2.5, 5, 10):
        if span / (step * factor) <= target:
            step *= factor
            break
    first = math.floor(low / step) * step
    last = math.ceil(high / step) * step
    ticks, value = [], first
    while value <= last + step / 2:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _text(x, y, content, size=12, fill=None, weight='normal', anchor='start', extra=''):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{_FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill or _hex("body")}" text-anchor="{anchor}" {extra}>'
        f'{escape(str(content))}</text>'
    )


def _header(spec, parts, legend_names=None):
    """Title, subtitle, unit and (for several series) the legend. Returns the
    y where the plot may start."""
    y = 36
    if spec['title']:
        parts.append(_text(24, y, _clip(spec['title'], 70), 17, _hex('ink'), 'bold'))
        y += 20
    if spec['subtitle']:
        parts.append(_text(24, y, _clip(spec['subtitle'], 90), 12, _hex('muted')))
        y += 18
    if spec['unit']:
        parts.append(_text(24, y, _clip(spec['unit'], 40), 12, _hex('muted')))
        y += 8
    if legend_names:
        x = 24
        y += 12
        for index, name in enumerate(legend_names):
            label = _clip(name, 22)
            if x + 22 + _text_width(label, 12) > WIDTH - 24:
                x, y = 24, y + 20
            parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{_color(index)}"/>')
            parts.append(_text(x + 18, y, label, 12, _hex('body')))
            x += 18 + _text_width(label, 12) + 18
        y += 8
    return y + 14


def _frame():
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="12" fill="#ffffff" '
        f'stroke="{_hex("rule")}" stroke-width="1"/>'
    )


def _value_range(series):
    values = [v for s in series for v in s['values'] if v is not None]
    low, high = min(values + [0]), max(values + [0])
    return _nice_ticks(low, high)


def _cartesian(spec, horizontal=False, line=False):
    labels, series = spec['labels'], spec['series']
    parts = [_frame()]
    legend = [s['name'] for s in series] if len(series) > 1 else None
    top = _header(spec, parts, legend)
    ticks = _value_range(series)
    low, high = ticks[0], ticks[-1]
    span = (high - low) or 1

    if horizontal:
        label_room = min(170, max(60, max((len(_clip(l, 22)) for l in labels), default=4) * 6.6 + 12))
        left, right, bottom = 24 + label_room, 40, 40
    else:
        left, right = 24 + max(len(format_value(t)) for t in ticks) * 6.8, 24
        many = len(labels) > 7 or max((len(l) for l in labels), default=0) > 9
        bottom = 78 if many else 50
    plot_w, plot_h = WIDTH - left - right, HEIGHT - top - bottom
    count = max(1, len(labels))

    def vx(value):
        return left + (value - low) / span * plot_w

    def vy(value):
        return top + plot_h - (value - low) / span * plot_h

    # Grid and value axis.
    for tick in ticks:
        if horizontal:
            x = vx(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
            parts.append(_text(x, top + plot_h + 18, format_value(tick), 11, _hex('muted'), anchor='middle'))
        else:
            y = vy(tick)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            parts.append(_text(left - 8, y + 4, format_value(tick), 11, _hex('muted'), anchor='end'))
    zero_x, zero_y = vx(0), vy(0)
    if horizontal:
        parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="#9ca3af" stroke-width="1.2"/>')
    else:
        parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#9ca3af" stroke-width="1.2"/>')

    group = (plot_h if horizontal else plot_w) / count
    show_values = count * len(series) <= 14

    if line:
        step = plot_w / count
        for s_index, s in enumerate(series):
            color = _color(s_index)
            points = [
                (left + step * (i + 0.5), vy(v)) if v is not None else None
                for i, v in enumerate(s['values'])
            ]
            run = []
            segments = []
            for point in points + [None]:
                if point is None:
                    if run:
                        segments.append(run)
                    run = []
                else:
                    run.append(point)
            for segment in segments:
                path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in segment)
                if len(series) == 1 and len(segment) > 1:
                    area = f'{segment[0][0]:.1f},{zero_y:.1f} {path} {segment[-1][0]:.1f},{zero_y:.1f}'
                    parts.append(f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>')
                if len(segment) > 1:
                    parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
            for (x, y), value in ((p, v) for p, v in zip(points, s['values']) if p):
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>')
                if show_values and len(series) == 1:
                    parts.append(_text(x, y - 11, format_value(value), 11, _hex('ink'), 'bold', 'middle'))
    else:
        bar = min(56, group * 0.72 / len(series))
        for g in range(count):
            for s_index, s in enumerate(series):
                value = s['values'][g] if g < len(s['values']) else None
                if value is None:
                    continue
                color = _color(s_index)
                offset = (group - bar * len(series)) / 2 + bar * s_index
                if horizontal:
                    y = top + group * g + offset
                    x0, x1 = sorted((vx(0), vx(value)))
                    parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1):.1f}" height="{bar - 2:.1f}" rx="3" fill="{color}"/>')
                    if show_values:
                        end = x1 + 6 if value >= 0 else x0 - 6
                        parts.append(_text(end, y + bar / 2 + 3, format_value(value), 11, _hex('ink'), 'bold', 'start' if value >= 0 else 'end'))
                else:
                    x = left + group * g + offset
                    y0, y1 = sorted((vy(0), vy(value)))
                    parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar - 2:.1f}" height="{max(y1 - y0, 1):.1f}" rx="3" fill="{color}"/>')
                    if show_values:
                        label_y = y0 - 6 if value >= 0 else y1 + 14
                        parts.append(_text(x + (bar - 2) / 2, label_y, format_value(value), 11, _hex('ink'), 'bold', 'middle'))

    # Category labels.
    for g, label in enumerate(labels):
        if horizontal:
            parts.append(_text(left - 8, top + group * g + group / 2 + 4, _clip(label, 22), 12, _hex('body'), anchor='end'))
        else:
            cx = left + group * (g + 0.5)
            if bottom > 60:
                parts.append(_text(cx, top + plot_h + 16, _clip(label, 16), 12, _hex('body'), anchor='end',
                                   extra=f'transform="rotate(-35 {cx:.1f} {top + plot_h + 16:.1f})"'))
            else:
                parts.append(_text(cx, top + plot_h + 20, _clip(label, 14), 12, _hex('body'), anchor='middle'))
    parts.append('</svg>')
    return ''.join(parts)


def _arc(cx, cy, r_out, r_in, start, end):
    large = 1 if end - start > math.pi else 0
    x0, y0 = cx + r_out * math.cos(start), cy + r_out * math.sin(start)
    x1, y1 = cx + r_out * math.cos(end), cy + r_out * math.sin(end)
    if r_in <= 0:
        return f'M {cx:.1f},{cy:.1f} L {x0:.1f},{y0:.1f} A {r_out:.1f},{r_out:.1f} 0 {large} 1 {x1:.1f},{y1:.1f} Z'
    x2, y2 = cx + r_in * math.cos(end), cy + r_in * math.sin(end)
    x3, y3 = cx + r_in * math.cos(start), cy + r_in * math.sin(start)
    return (
        f'M {x0:.1f},{y0:.1f} A {r_out:.1f},{r_out:.1f} 0 {large} 1 {x1:.1f},{y1:.1f} '
        f'L {x2:.1f},{y2:.1f} A {r_in:.1f},{r_in:.1f} 0 {large} 0 {x3:.1f},{y3:.1f} Z'
    )


def _pie(spec, donut=False):
    parts = [_frame()]
    top = _header(spec, parts)
    values = [(l, v) for l, v in zip(spec['labels'], spec['series'][0]['values']) if v]
    total = sum(v for _, v in values) or 1
    radius = min((HEIGHT - top - 24) / 2, 150)
    cx, cy = 24 + radius + 10, top + (HEIGHT - top - 12) / 2
    angle = -math.pi / 2
    for index, (label, value) in enumerate(values):
        sweep = value / total * 2 * math.pi
        if len(values) == 1:   # a full circle cannot be drawn as one arc
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{_color(index)}"/>')
            if donut:
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius * 0.55:.1f}" fill="#ffffff"/>')
        else:
            parts.append(
                f'<path d="{_arc(cx, cy, radius, radius * 0.55 if donut else 0, angle, angle + sweep)}" '
                f'fill="{_color(index)}" stroke="#ffffff" stroke-width="2"/>'
            )
        share = value / total * 100
        if share >= 6:
            mid = angle + sweep / 2
            reach = radius * (0.775 if donut else 0.64)
            parts.append(_text(cx + reach * math.cos(mid), cy + reach * math.sin(mid) + 4,
                               f'{share:.0f}%' if share >= 9.5 else f'{share:.1f}%', 12, '#ffffff', 'bold', 'middle'))
        angle += sweep

    legend_x = cx + radius + 40
    row = min(26, (HEIGHT - top - 20) / max(1, len(values)))
    y = top + 8
    for index, (label, value) in enumerate(values):
        parts.append(f'<rect x="{legend_x:.1f}" y="{y - 10:.1f}" width="12" height="12" rx="3" fill="{_color(index)}"/>')
        parts.append(_text(legend_x + 20, y, f'{_clip(label, 26)}', 12, _hex('body')))
        parts.append(_text(WIDTH - 24, y, f'{format_value(value)}  ·  {value / total * 100:.1f}%', 12, _hex('muted'), anchor='end'))
        y += row
    parts.append('</svg>')
    return ''.join(parts)


def chart_svg(spec):
    kind = spec['type']
    if kind in ('pie', 'donut'):
        return _pie(spec, donut=kind == 'donut')
    return _cartesian(spec, horizontal=kind == 'hbar', line=kind == 'line')


def chart_png(spec, scale=2.0):
    """The chart as PNG bytes (WIDTH*scale by HEIGHT*scale pixels)."""
    import pymupdf
    document = pymupdf.open(stream=chart_svg(spec).encode('utf-8'), filetype='svg')
    try:
        return document[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')
    finally:
        document.close()


def chart_ratio():
    return HEIGHT / WIDTH
