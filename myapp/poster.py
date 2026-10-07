"""Posters whose words come out exactly right.

The image model (FLUX.2 Klein, 4 steps) paints good artwork but cannot spell:
a brand name, an email address, a phone number or a price comes back garbled.
So a poster request is split in two:

1. the words the user wants on it (brand name, tagline, features, offer and
   contact details) are lifted out of the prompt - email, phone and website by
   pattern, the rest by a short model call that may only copy the user's own
   words, checked again here;
2. FLUX paints just the background artwork, told to leave out all text;
3. Pillow letters the exact words on top in a fixed, readable layout with the
   bundled Poppins font (SIL Open Font License, myapp/fonts/OFL.txt).

Anything that is not a poster - "a cat on the moon" - never comes here.
"""
import io
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from myapp import image_generation

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent / 'fonts'
MAX_FEATURES = 6

# What the user wants written that must be exact.
EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
PHONE_RE = re.compile(r'(?<![\w@])\+?\d[\d \-]{6,16}\d(?![\w@])')
WEBSITE_RE = re.compile(
    r'\b(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*'
    r'\.(?:com|in|org|net|io|ai|co|app|dev|edu|info|biz|me|xyz|store|shop|online|tech|site|live|in)'
    r'(?:\.[a-z]{2})?(?:/[^\s,)]*)?(?![\w@])',
    re.IGNORECASE,
)
PRICE_RE = re.compile(
    r'(?:₹|\brs\.?|\binr|\$|\busd|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?/\s?(?:month|mo|year|yr|day|week))?',
    re.IGNORECASE,
)
POSTER_CUE_RE = re.compile(
    r'\b(?:poster|flyer|flier|banner|brochure|pamphlet|leaflet|advert(?:isement)?|ad creative|creative|'
    r'promo(?:tional)?|announcement|social media post|instagram|insta|facebook|linkedin|post for|'
    r'brand(?:ing)?|business card|hoarding|billboard|thumbnail)\b',
    re.IGNORECASE,
)
_PICTURE_RE = re.compile(r'\b(?:image|picture|pic|photo|poster|flyer|banner|graphic|design|creative|post)\b', re.I)
# Pillow without libraqm cannot join Indian-script letters correctly, so such
# posters stay with the ordinary image path rather than come out broken.
_COMPLEX_SCRIPT_RE = re.compile('[ऀ-෿؀-ۿ]')


def _contact_details(prompt):
    emails = []
    for found in EMAIL_RE.findall(prompt):
        if found.lower() not in (e.lower() for e in emails):
            emails.append(found.rstrip('.'))
    without_emails = EMAIL_RE.sub(' ', prompt)
    phones = []
    for found in PHONE_RE.findall(without_emails):
        digits = re.sub(r'\D', '', found)
        if 8 <= len(digits) <= 13 and found.strip() not in phones:
            phones.append(found.strip())
    websites = []
    for found in WEBSITE_RE.findall(without_emails):
        found = found.rstrip('.')
        if found.lower() not in (w.lower() for w in websites):
            websites.append(found)
    return emails[:2], phones[:2], websites[:2]


def wants_poster(prompt):
    """True for a picture request carrying words that must be printed exactly:
    contact details or a price, together with a poster/brand cue (or several
    kinds of details at once)."""
    text = prompt or ''
    if not _PICTURE_RE.search(text):
        return False
    emails, phones, websites = _contact_details(text)
    kinds = sum(bool(x) for x in (emails, phones, websites, PRICE_RE.search(text)))
    return bool(kinds and (POSTER_CUE_RE.search(text) or kinds >= 2))


@dataclass
class PosterSpec:
    title: str = ''
    tagline: str = ''
    features: list = field(default_factory=list)
    offer: str = ''
    cta: str = ''
    emails: list = field(default_factory=list)
    phones: list = field(default_factory=list)
    websites: list = field(default_factory=list)
    scene: str = ''

    def words(self):
        return [self.title, self.tagline, self.offer, self.cta, *self.features, *self.emails, *self.phones, *self.websites]

    def has_text(self):
        return any(self.words())


_SPEC_SYSTEM = (
    "You prepare the exact words for a poster from the user's request. Copy every piece of text EXACTLY as the "
    "user wrote it - same spelling, capitals, numbers and symbols. Never invent, translate, shorten or add words; "
    "leave a field empty when the user did not write it. Reply with ONLY a JSON object:\n"
    '{"title": "the brand or main headline", "tagline": "a subtitle the user wrote, or empty", '
    '"features": ["each feature or service the user listed, max 6"], '
    '"offer": "the price or offer line, e.g. Plans starting at ₹99, or empty", '
    '"cta": "a call to action only if the user wrote one, else empty", '
    '"scene": "the background artwork only: style, colours, mood, lighting and objects the user asked for, in '
    'English, under 45 words, with NO words, names, prices or contact details, and nothing that would carry '
    'writing such as screens, user interfaces, documents, signs or labels"}'
)


def _tokens(text):
    return re.findall(r'\w+', (text or '').lower())


def _grounded(value, prompt):
    """A model-copied value only counts if it really is the user's own words:
    every number exact, and nearly every word present in the prompt."""
    value = re.sub(r'\s+', ' ', str(value or '')).strip(' "\'“”')
    if not value or len(value) > 120:
        return ''
    lowered = prompt.lower()
    for number in re.findall(r'\d+', value):
        if number not in prompt:
            return ''
    words = _tokens(value)
    if not words:
        return ''
    prompt_words = set(_tokens(prompt))
    present = sum(1 for w in words if w in prompt_words or w in lowered)
    return value if present / len(words) >= 0.8 else ''


def _fallback_spec(prompt):
    """Pattern-only reading, for when the model call is unavailable."""
    spec = PosterSpec()
    brand = re.search(
        r'\bbrand(?:\s+name)?\s*(?:is|:|called|named)?\s*["“]?([^,."”\n]{2,80})', prompt, re.IGNORECASE,
    )
    quoted = re.search(r'["“]([^"”]{2,60})["”]', prompt)
    head = brand.group(1) if brand else (quoted.group(1) if quoted else '')
    if head:
        parts = re.split(r'\s+[–—-]\s+|\s*:\s+', head, maxsplit=1)
        title = re.split(r'\s*,?\s*\b(?:featuring|with|for|that|which|including)\b', parts[0], 1, flags=re.I)[0]
        spec.title = title.strip()
        if len(parts) > 1:
            spec.tagline = re.split(r'\s*,?\s*\b(?:featuring|with|including)\b', parts[1], 1, flags=re.I)[0].strip()
    listed = re.search(
        r'\b(?:featuring|features?|including|offering|services?)\s*:?\s+(.+?)(?:,\s*(?:with|and with)\b|\.\s|\.$|$)',
        prompt, re.IGNORECASE,
    )
    if listed:
        items = re.split(r',\s*|\s+and\s+', listed.group(1))
        spec.features = [i.strip(' .') for i in items if 1 < len(i.strip(' .')) <= 60][:MAX_FEATURES]
    price = PRICE_RE.search(prompt)
    if price:
        lead = re.search(
            r'\b((?:plans?|prices?|pricing|starting|starts|from|only|just|at)\b[^,.;]{0,30}?)' + re.escape(price.group(0)),
            prompt, re.IGNORECASE,
        )
        offer = (lead.group(1) + price.group(0)) if lead else price.group(0)
        spec.offer = offer[0].upper() + offer[1:]
    return spec


_TEXT_ASK_RE = re.compile(
    r'[^,.;]*\b(?:typography|typeface|text|font|lettering|readable|branding|brand|logo|hierarchy|headline|'
    r'caption|label|title|words?|instagram|portrait size|landscape size|\d{3,5}\s*[x×]\s*\d{3,5})\b[^,.;]*',
    re.IGNORECASE,
)


def _scrub_scene(scene):
    """The artwork description without anything that asks for words on the picture."""
    scene = _TEXT_ASK_RE.sub('', scene or '')
    scene = re.sub(r'\s*,(\s*,)+', ',', scene)
    return re.sub(r'\s{2,}', ' ', scene).strip(' ,.;')


def _scene_fallback(prompt):
    style = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', prompt)
        if re.search(r'\b(?:background|theme|style|colou?rs?|glow\w*|futuristic|elements|mood|lighting)\b', s, re.I)
        and not (EMAIL_RE.search(s) or PRICE_RE.search(s))
    ]
    scene = _scrub_scene(' '.join(style)[:300])
    return scene or 'premium abstract futuristic technology background with soft glowing light'


def _capitalised(text):
    """A line starting with a capital, the only change ever made to the user's words."""
    return text[:1].upper() + text[1:] if text[:1].islower() else text


def extract_spec(prompt):
    emails, phones, websites = _contact_details(prompt)
    spec = PosterSpec(emails=emails, phones=phones, websites=websites)
    reply = None
    try:
        from myapp import ai_chat
        reply = ai_chat.complete_json(_SPEC_SYSTEM, prompt[:3000], model_key='quick', max_tokens=700, timeout=20.0, temperature=0.1)
    except Exception as exc:
        logger.info('Poster text could not be read by the model (%s); using patterns', exc.__class__.__name__)
    if isinstance(reply, dict):
        spec.title = _grounded(reply.get('title'), prompt)
        spec.tagline = _grounded(reply.get('tagline'), prompt)
        spec.offer = _grounded(reply.get('offer'), prompt)
        spec.cta = _grounded(reply.get('cta'), prompt)
        raw_features = reply.get('features') if isinstance(reply.get('features'), list) else []
        spec.features = [f for f in (_grounded(x, prompt) for x in raw_features) if f][:MAX_FEATURES]
        scene = str(reply.get('scene') or '').strip()
        # The scene goes to the image model: it must not carry the words back in.
        if scene and not (EMAIL_RE.search(scene) or PRICE_RE.search(scene) or WEBSITE_RE.search(scene)):
            spec.scene = _scrub_scene(scene)[:400]
    fallback = _fallback_spec(prompt)
    for name in ('title', 'tagline', 'offer'):
        if not getattr(spec, name):
            setattr(spec, name, getattr(fallback, name))
    if not spec.features:
        spec.features = fallback.features
    # "Brand – Tagline" written as one heading becomes a title and a tagline.
    if spec.title and not spec.tagline:
        parts = re.split(r'\s+[–—|-]\s+', spec.title, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            spec.title, spec.tagline = parts[0].strip(), parts[1].strip()
    # Contact details live in their own panel; a line that repeats them goes.
    contact_bits = [c.lower() for c in (*emails, *websites)] + [re.sub(r'\D', '', p) for p in phones]

    def repeats_contact(text):
        lowered, digits = text.lower(), re.sub(r'\D', '', text)
        return any(bit and (bit in lowered or (bit.isdigit() and bit in digits)) for bit in contact_bits)

    if spec.cta and repeats_contact(spec.cta):
        spec.cta = ''
    if spec.offer and repeats_contact(spec.offer):
        spec.offer = re.split(r'\s*[,;·]\s*|\s+(?:call|contact|visit|email|whatsapp)\b', spec.offer, 1, flags=re.I)[0]
        if not PRICE_RE.search(spec.offer):
            spec.offer = ''
    spec.features = [f for f in spec.features if not repeats_contact(f)]
    for name in ('tagline', 'offer', 'cta'):
        setattr(spec, name, _capitalised(getattr(spec, name)))
    spec.features = [_capitalised(f) for f in spec.features]
    if spec.tagline and spec.title:
        if spec.tagline.lower() == spec.title.lower():
            spec.tagline = ''
        elif spec.title.lower().endswith(spec.tagline.lower()):
            spec.title = re.sub(r'[\s:–—|-]+$', '', spec.title[: -len(spec.tagline)]).strip() or spec.title
    if not spec.scene:
        spec.scene = _scene_fallback(prompt)
    return spec


def background_prompt(spec):
    return (
        f'{spec.scene.rstrip(". ")}. Background artwork for a poster, rich detail in the middle, calmer darker '
        'areas at the top and bottom. No text, no letters, no words, no numbers, no logos, no watermark.'
    )


def output_size(prompt, flux_size):
    """The poster's final pixel size: exactly what was asked for ("1080×1350"),
    otherwise the image model's shape scaled to a 1080-pixel short side."""
    match = image_generation._PIXEL_SIZE_RE.search(prompt or '')
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        if 300 <= width <= 4096 and 300 <= height <= 4096 and 0.25 <= width / height <= 4:
            scale = min(1.0, 2160 / max(width, height))
            return round(width * scale), round(height * scale)
    w, h = flux_size
    scale = 1080 / min(w, h)
    return round(w * scale), round(h * scale)


# ---- lettering ----

@lru_cache(maxsize=96)
def _font(weight, size):
    return ImageFont.truetype(str(FONT_DIR / f'Poppins-{weight}.ttf'), max(8, int(round(size))))


@lru_cache(maxsize=1)
def _missing_glyph_mask():
    return _glyph_mask('\U000F0000')


@lru_cache(maxsize=512)
def _glyph_mask(ch):
    im = Image.new('L', (90, 90))
    ImageDraw.Draw(im).text((10, 5), ch, font=_font('Medium', 50), fill=255)
    return im.tobytes()


def printable(text):
    """``text`` without characters the font cannot draw (emoji and the like)."""
    kept = ''.join(ch for ch in text if ch.isspace() or _glyph_mask(ch) != _missing_glyph_mask())
    return re.sub(r'\s{2,}', ' ', kept).strip()


def can_letter(spec):
    if not features.check('raqm') and any(_COMPLEX_SCRIPT_RE.search(w) for w in spec.words() if w):
        return False
    return True


def _wrap(text, font, max_width, max_lines):
    """Lines of ``text`` fitting ``max_width``; None when it needs more lines."""
    words = text.split()
    lines, line = [], ''
    for word in words:
        trial = f'{line} {word}'.strip()
        if font.getlength(trial) <= max_width:
            line = trial
            continue
        if line:
            lines.append(line)
        if font.getlength(word) > max_width:
            return None
        line = word
    if line:
        lines.append(line)
    return lines if len(lines) <= max_lines else None


def _fit(text, weight, size, min_size, max_width, max_lines):
    """The largest font (from ``size`` down to ``min_size``) that fits."""
    size = float(size)
    while size >= min_size:
        font = _font(weight, size)
        lines = _wrap(text, font, max_width, max_lines)
        if lines:
            return font, lines
        size *= 0.93
    font = _font(weight, min_size)
    return font, _wrap(text, font, max_width, 6) or [text]


def _line_height(font):
    ascent, descent = font.getmetrics()
    return ascent + descent


def _accent_colour(image):
    """The artwork's main vivid colour (the hue covering the most of it, not
    one bright speck), lifted so it reads on dark glass. Sky blue when the
    artwork is mostly grey."""
    small = image.convert('RGB').resize((64, 64))
    bins = {}
    for (r, g, b), (h, sat, val) in zip(small.getdata(), small.convert('HSV').getdata()):
        if sat < 90 or val < 70:
            continue
        weight = sat * val
        total = bins.setdefault(h * 12 // 256, [0, 0, 0, 0])
        total[0] += r * weight
        total[1] += g * weight
        total[2] += b * weight
        total[3] += weight
    if not bins or max(t[3] for t in bins.values()) < 64 * 64 * 255 * 255 * 0.004:
        return (56, 189, 248)
    r, g, b, weight = max(bins.values(), key=lambda t: t[3])
    colour = (r / weight, g / weight, b / weight)
    lift = 235 / max(colour)
    return tuple(min(255, int(c * lift)) for c in colour)


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _cover(image, size):
    width, height = size
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.convert('RGB').resize((max(width, round(src_w * scale)), max(height, round(src_h * scale))), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _shade(canvas):
    """Darken the top and bottom so white lettering always reads."""
    width, height = canvas.size
    column = Image.new('L', (1, height))
    for y in range(height):
        t = y / height
        top = max(0.0, 1 - t / 0.45) * 225
        bottom = max(0.0, (t - 0.45) / 0.55) * 225
        column.putpixel((0, y), int(max(top, bottom) + 35))
    shade = Image.new('RGBA', (width, height), (4, 7, 18, 0))
    shade.putalpha(column.resize((width, height)))
    canvas.alpha_composite(shade)


def _glass(canvas, box, radius, accent, s, fill_alpha=150):
    """A frosted panel: the artwork behind it blurred, tinted dark, edged in the accent."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    region = canvas.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(14 * s))
    mask = Image.new('L', region.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, region.width - 1, region.height - 1), radius=radius, fill=255)
    canvas.paste(region, (x0, y0), mask)
    layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(9, 13, 28, fill_alpha),
                           outline=accent + (150,), width=max(1, int(2 * s)))
    canvas.alpha_composite(layer)


def _icon(draw, kind, cx, cy, r, colour, s):
    w = max(2, int(round(2.6 * s)))
    if kind == 'check':
        draw.line([(cx - r * .45, cy + r * .02), (cx - r * .1, cy + r * .38), (cx + r * .5, cy - r * .35)], fill=colour, width=w, joint='curve')
    elif kind == 'email':
        box = (cx - r * .55, cy - r * .38, cx + r * .55, cy + r * .38)
        draw.rounded_rectangle(box, radius=r * .08, outline=colour, width=w)
        draw.line([(box[0], box[1]), (cx, cy + r * .08), (box[2], box[1])], fill=colour, width=w, joint='curve')
    elif kind == 'phone':
        draw.rounded_rectangle((cx - r * .32, cy - r * .55, cx + r * .32, cy + r * .55), radius=r * .12, outline=colour, width=w)
        draw.line([(cx - r * .1, cy + r * .38), (cx + r * .1, cy + r * .38)], fill=colour, width=w)
    elif kind == 'web':
        draw.ellipse((cx - r * .55, cy - r * .55, cx + r * .55, cy + r * .55), outline=colour, width=w)
        draw.ellipse((cx - r * .24, cy - r * .55, cx + r * .24, cy + r * .55), outline=colour, width=w)
        draw.line([(cx - r * .55, cy), (cx + r * .55, cy)], fill=colour, width=w)


def _plan(spec, width, height, s):
    """Fonts, lines and heights for every block at scale ``s``; None if it
    does not fit on the canvas."""
    pad = 64 * s
    inner = width - 2 * pad
    plan = {'s': s, 'pad': pad}
    y = pad * 0.9
    if spec.title:
        font, lines = _fit(spec.title, 'Bold', 128 * s, 54 * s, inner, 2)
        plan['title'] = (font, lines, y)
        y += _line_height(font) * len(lines) * 0.98
    if spec.tagline:
        font, lines = _fit(spec.tagline, 'Medium', 46 * s, 26 * s, inner, 2)
        y += 6 * s
        plan['tagline'] = (font, lines, y)
        y += _line_height(font) * len(lines)
    if spec.title or spec.tagline:
        plan['rule'] = y + 18 * s
        y += 30 * s
    top_end = y

    bottom = height - pad * 0.9
    contacts = [('email', e) for e in spec.emails] + [('phone', p) for p in spec.phones] + [('web', w) for w in spec.websites]
    if contacts:
        row_h = 60 * s
        icon_d = 46 * s
        text_w = inner - 2 * 28 * s - icon_d - 20 * s
        rows = []
        for kind, value in contacts:
            font, lines = _fit(value, 'Medium', 32 * s, 18 * s, text_w, 1)
            rows.append((kind, value, font))
        panel_h = len(rows) * row_h + 2 * 22 * s
        plan['contacts'] = (rows, bottom - panel_h, panel_h, row_h, icon_d)
        bottom -= panel_h + 26 * s
    offer_text = ' · '.join(t for t in (spec.offer, spec.cta) if t)
    if offer_text:
        font, lines = _fit(offer_text, 'Bold', 40 * s, 22 * s, inner - 80 * s, 1)
        pill_h = _line_height(font) + 34 * s
        plan['offer'] = (font, offer_text, bottom - pill_h, pill_h)
        bottom -= pill_h + 26 * s
    if spec.features:
        columns = 2 if len(spec.features) >= 3 and width >= height * 0.6 else 1
        gap = 18 * s
        col_w = (inner - gap * (columns - 1)) / columns
        dot = 32 * s
        chips = []
        for text in spec.features:
            font, lines = _fit(text, 'SemiBold', 30 * s, 20 * s, col_w - dot - 3 * 18 * s, 2)
            chips.append((font, lines, _line_height(font) * len(lines) + 30 * s))
        rows = [chips[i:i + columns] for i in range(0, len(chips), columns)]
        row_heights = [max(c[2] for c in row) for row in rows]
        grid_h = sum(row_heights) + gap * (len(rows) - 1)
        plan['features'] = (rows, row_heights, bottom - grid_h, columns, col_w, gap, dot)
        bottom -= grid_h
    plan['fits'] = bottom >= top_end + 24 * s
    return plan


def compose(background, spec, size):
    """The finished poster as JPEG bytes."""
    width, height = size
    canvas = _cover(background, size).convert('RGBA')
    accent = _accent_colour(canvas)
    _shade(canvas)
    base = min(width, height) / 1080
    plan = None
    for factor in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.55):
        plan = _plan(spec, width, height, base * factor)
        if plan['fits']:
            break
    s, pad = plan['s'], plan['pad']
    light_accent = _mix(accent, (255, 255, 255), 0.55)

    if 'title' in plan:
        font, lines, y = plan['title']
        glow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        ty = y
        for line in lines:
            glow_draw.text((width / 2, ty), line, font=font, fill=accent + (200,), anchor='mt')
            ty += _line_height(font) * 0.98
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(16 * s)))
        draw = ImageDraw.Draw(canvas)
        ty = y
        for line in lines:
            draw.text((width / 2, ty), line, font=font, fill=(255, 255, 255, 255), anchor='mt')
            ty += _line_height(font) * 0.98
    draw = ImageDraw.Draw(canvas)
    if 'tagline' in plan:
        font, lines, y = plan['tagline']
        for line in lines:
            draw.text((width / 2, y), line, font=font, fill=light_accent + (255,), anchor='mt')
            y += _line_height(font)
    if 'rule' in plan:
        y = plan['rule']
        draw.rounded_rectangle((width / 2 - 70 * s, y, width / 2 + 70 * s, y + 6 * s), radius=3 * s, fill=accent + (255,))

    if 'features' in plan:
        rows, row_heights, y, columns, col_w, gap, dot = plan['features']
        for row, row_h in zip(rows, row_heights):
            row_w = len(row) * col_w + gap * (len(row) - 1)
            x = (width - row_w) / 2
            for font, lines, _ in row:
                _glass(canvas, (x, y, x + col_w, y + row_h), 18 * s, accent, s, 140)
                draw = ImageDraw.Draw(canvas)
                cx, cy = x + 18 * s + dot / 2, y + row_h / 2
                draw.ellipse((cx - dot / 2, cy - dot / 2, cx + dot / 2, cy + dot / 2), fill=accent + (255,))
                _icon(draw, 'check', cx, cy, dot / 2, (8, 12, 26, 255), s)
                text_h = _line_height(font) * len(lines)
                ty = y + (row_h - text_h) / 2
                for line in lines:
                    draw.text((x + 18 * s + dot + 16 * s, ty), line, font=font, fill=(255, 255, 255, 255))
                    ty += _line_height(font)
                x += col_w + gap
            y += row_h + gap

    if 'offer' in plan:
        font, text, y, pill_h = plan['offer']
        text_w = font.getlength(text)
        pill_w = text_w + 80 * s
        x0 = (width - pill_w) / 2
        pill = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        pill_draw = ImageDraw.Draw(pill)
        pill_draw.rounded_rectangle((x0, y, x0 + pill_w, y + pill_h), radius=pill_h / 2, fill=accent + (255,))
        glow = pill.filter(ImageFilter.GaussianBlur(14 * s))
        canvas.alpha_composite(glow)
        canvas.alpha_composite(pill)
        draw = ImageDraw.Draw(canvas)
        draw.text((width / 2, y + pill_h / 2), text, font=font, fill=(8, 12, 26, 255), anchor='mm')

    if 'contacts' in plan:
        rows, y, panel_h, row_h, icon_d = plan['contacts']
        _glass(canvas, (pad, y, width - pad, y + panel_h), 24 * s, accent, s, 165)
        draw = ImageDraw.Draw(canvas)
        widest = max(icon_d + 20 * s + font.getlength(value) for _, value, font in rows)
        x = max(pad + 28 * s, (width - widest) / 2)
        ry = y + 22 * s
        for kind, value, font in rows:
            cy = ry + row_h / 2
            cx = x + icon_d / 2
            draw.ellipse((cx - icon_d / 2, cy - icon_d / 2, cx + icon_d / 2, cy + icon_d / 2), fill=accent + (255,))
            _icon(draw, kind, cx, cy, icon_d / 2, (8, 12, 26, 255), s)
            draw.text((x + icon_d + 20 * s, cy), value, font=font, fill=(255, 255, 255, 255), anchor='lm')
            ry += row_h

    output = io.BytesIO()
    canvas.convert('RGB').save(output, format='JPEG', quality=93, optimize=True)
    return output.getvalue()


def make_poster(prompt):
    """The finished poster as a GeneratedImage, or None when this request is
    not a poster that can be lettered here (the caller then draws it the usual
    way). Image-service failures are raised as ImageGenerationError."""
    if not wants_poster(prompt):
        return None
    if not features.check('raqm') and _COMPLEX_SCRIPT_RE.search(prompt):
        return None
    spec = extract_spec(prompt)
    for name in ('title', 'tagline', 'offer', 'cta'):
        setattr(spec, name, printable(getattr(spec, name)))
    spec.features = [f for f in (printable(x) for x in spec.features) if f]
    if not spec.has_text() or not can_letter(spec):
        return None
    flux_size = image_generation.resolve_dimensions(prompt)
    artwork = image_generation.generate_image(background_prompt(spec), size=flux_size)
    try:
        with Image.open(io.BytesIO(artwork.content)) as opened:
            opened.load()
            content = compose(opened, spec, output_size(prompt, flux_size))
    except Exception:
        logger.exception('Could not letter the poster; drawing it the usual way')
        return None
    return image_generation.GeneratedImage(content=content, extension='jpg')
