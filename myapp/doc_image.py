"""An image from the chat (a generated poster, an uploaded photo) placed in a
PDF, Word or PowerPoint file - "add this image inside a pdf".

The picture is stored with the AIGeneratedFile row (content = PREFIX + its
data: URI) and the file is built when it is downloaded, like every other
generated file. Nothing is redrawn: the image goes in as it is, fitted to the
page without cropping.
"""
import base64
import binascii
import io
import re

from PIL import Image, ImageOps

PREFIX = 'image-file:'
EXTENSIONS = ('pdf', 'docx', 'pptx')
CONTENT_TYPES = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
}
_DATA_URI_RE = re.compile(r'^data:image/[a-z+.-]+;base64,(?P<data>[A-Za-z0-9+/=\s]+)$', re.IGNORECASE)


def stored_content(data_uri):
    return PREFIX + data_uri


def is_image_file(content):
    return (content or '').startswith(PREFIX)


def _picture(content):
    """(JPEG or PNG bytes, extension, width, height) of the stored image."""
    match = _DATA_URI_RE.match(content[len(PREFIX):].strip())
    if not match:
        raise ValueError('The stored image is not a data URI.')
    try:
        raw = base64.b64decode(match.group('data'), validate=False)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('The stored image could not be decoded.') from exc
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened)
        image.load()
    out = io.BytesIO()
    if image.mode in ('RGBA', 'LA', 'P'):
        image.convert('RGBA').save(out, format='PNG', optimize=True)
        extension = 'png'
    else:
        image.convert('RGB').save(out, format='JPEG', quality=95)
        extension = 'jpeg'
    return out.getvalue(), extension, image.width, image.height


def _pdf(picture, width, height):
    import fitz
    page_w = 595.0                      # A4 width; the page takes the picture's shape
    page_h = page_w * height / width
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    page.insert_image(page.rect, stream=picture, keep_proportion=True)
    doc.set_metadata({'title': 'Image'})
    return doc.tobytes(garbage=3, deflate=True)


def _docx(picture, width, height):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Emu, Inches
    document = Document()
    section = document.sections[0]
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(section, side, Inches(0.6))
    usable_w = section.page_width - section.left_margin - section.right_margin
    usable_h = section.page_height - section.top_margin - section.bottom_margin - Inches(0.3)
    shown_w = min(usable_w, int(usable_h * width / height))
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(io.BytesIO(picture), width=Emu(shown_w))
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _pptx(picture, width, height):
    from pptx import Presentation
    from pptx.util import Emu, Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    margin = Inches(0.3)
    box_w, box_h = prs.slide_width - 2 * margin, prs.slide_height - 2 * margin
    scale = min(box_w / width, box_h / height)
    shown_w, shown_h = int(width * scale), int(height * scale)
    slide.shapes.add_picture(
        io.BytesIO(picture), Emu((prs.slide_width - shown_w) // 2), Emu((prs.slide_height - shown_h) // 2),
        width=Emu(shown_w), height=Emu(shown_h),
    )
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


def render(file_name, content):
    """The file's bytes for the download."""
    extension = file_name.rsplit('.', 1)[-1].lower()
    picture, _, width, height = _picture(content)
    builder = {'pdf': _pdf, 'docx': _docx, 'pptx': _pptx}[extension]
    return builder(picture, width, height)
