"""NVIDIA FLUX image generation and editing for the AI chat."""

import base64
import binascii
import io
import re
from dataclasses import dataclass

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from myapp.provider_keys import get_key as _get_key


FLUX_API_URL = (
    "https://ai.api.nvidia.com/v1/genai/"
    "black-forest-labs/flux.2-klein-4b"
)
REQUEST_TIMEOUT_SECONDS = 120
MAX_PROMPT_CHARS = 10_000
# FLUX.2 Klein's hosted NIM endpoint 400s on long, heavily-detailed prompts
# well under MAX_PROMPT_CHARS above (that limit only guards against genuinely
# absurd input). Rather than surface that as an error, a prompt longer than
# this gets shortened to its leading sentences before being sent — almost
# always still enough to carry the actual subject/style/composition intent.
FLUX_SAFE_PROMPT_CHARS = 480

# Every generated image used to be a hardcoded 1024x1024 square, whatever was
# asked for. Three of the four newest reports were exactly that: #48 wanted a
# "wallpaper 4k resolution", #47 a "size 9.12" greeting, #45 an Instagram post
# — all three got a square back and were reported as wrong.
#
# The ceiling is not a guess: the endpoint rejects anything larger with
# "Requested image size 1069056 exceeds supported image size 1062400", and the
# sizes below were confirmed against the live API to come back at exactly the
# requested dimensions. All are multiples of 32, which diffusion models want.
MAX_PIXELS = 1_062_400
DEFAULT_SIZE = (1024, 1024)

_ASPECT_SIZES = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "5:4": (1120, 896),
    "4:5": (896, 1120),
}

# Ordered: the more specific phrasing has to win, because a phone wallpaper is
# portrait while a plain "wallpaper" is a desktop one.
_ORIENTATION_CUES = (
    (r"\b(?:phone|mobile|iphone|android|smartphone)\s+(?:wallpaper|background|screen)\b", "9:16"),
    (r"\b(?:story|stories|reel|reels|short|shorts|status|snapchat|tiktok)\b", "9:16"),
    (r"\b(?:portrait|vertical|upright|full\s?screen\s+phone)\b", "9:16"),
    (r"\b(?:instagram|insta|ig|facebook|fb|linkedin)\s+(?:post|feed|creative)\b", "4:5"),
    (r"\b(?:wallpaper|desktop|laptop|monitor|widescreen|wide\s?screen)\b", "16:9"),
    (r"\b(?:banner|cover\s+(?:photo|image)|header|thumbnail|youtube|hoarding|billboard)\b", "16:9"),
    (r"\b(?:landscape|horizontal|panorama|panoramic)\b", "16:9"),
    (r"\b(?:poster|flyer|pamphlet|brochure|leaflet|invitation|invite|a4|certificate)\b", "3:4"),
    (r"\b(?:profile\s+(?:picture|pic|photo)|avatar|logo|icon|dp|display\s+picture)\b", "1:1"),
    (r"\b(?:square)\b", "1:1"),
)
_ORIENTATION_CUES = tuple(
    (re.compile(pattern, re.IGNORECASE), ratio) for pattern, ratio in _ORIENTATION_CUES
)

# "1920x1080", "1080 X 1920" — an exact pixel request states the ratio outright.
_PIXEL_SIZE_RE = re.compile(r"\b(\d{3,5})\s*[x×*]\s*(\d{3,5})\b", re.IGNORECASE)
# "16:9", "4/5". A bare dot is deliberately NOT accepted here: it would read
# "ChatGPT 5.6" as a 5:6 portrait.
_RATIO_RE = re.compile(r"\b(\d{1,2})\s*[:/]\s*(\d{1,2})\b")
# An unannounced "9:30" is a clock time far more often than an aspect ratio,
# and "the number 1/2" is a fraction. Without a size word to anchor it, only
# ratios people genuinely use for images count — "good morning image at 9:30
# am" was otherwise coming out as a portrait.
_BARE_RATIO_ALLOWLIST = frozenset({
    (1, 1), (3, 2), (2, 3), (4, 3), (3, 4), (5, 4), (4, 5),
    (16, 9), (9, 16), (16, 10), (10, 16), (21, 9), (5, 3), (3, 5),
})
# After an explicit size word a dot is safe, which is what makes report #47's
# "size 9.12" readable as the 9:12 (i.e. 3:4) portrait they wanted.
_SIZED_RATIO_RE = re.compile(
    r"\b(?:size|ratio|aspect|resolution|dimensions?)\b\D{0,10}(\d{1,2})\s*[.:/x]\s*(\d{1,2})\b",
    re.IGNORECASE,
)
_RATIO_SIZED_RE = re.compile(
    r"\b(\d{1,2})\s*[.:/x]\s*(\d{1,2})\s*(?:size|ratio|aspect|resolution|dimensions?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    extension: str


class ImageGenerationError(Exception):
    """A safe, user-facing failure from the image generation service."""

    def __init__(self, message, *, status_code=503, blocked=False, editing_unavailable=False):
        super().__init__(message)
        self.status_code = status_code
        # True when a backend refused the prompt itself (its content filter,
        # or a generic 400 that's most often the same thing in practice) —
        # generate_image() uses this to decide whether retrying on a
        # different backend with a cleaned-up prompt is worth attempting.
        self.blocked = blocked
        # True specifically when the connected backend cannot edit an
        # uploaded reference image at all (as opposed to rejecting this
        # particular prompt/image) — the caller can fall back to describing
        # the photo and generating a fresh image from text instead.
        self.editing_unavailable = editing_unavailable


def _safe_error(response):
    status_code = response.status_code
    try:
        detail = str(response.json().get("detail", "")).lower()
    except (ValueError, AttributeError):
        detail = ""
    if status_code == 422 and "expected: example_id" in detail:
        return ImageGenerationError(
            "Uploaded-image editing is unavailable on the connected image service. Please contact support to enable it.",
            status_code=503,
            editing_unavailable=True,
        )
    if status_code == 429:
        return ImageGenerationError(
            "The image-generation limit has been reached. Please wait and try again later.",
            status_code=429,
        )
    if status_code in (400, 413, 422):
        return ImageGenerationError(
            "The image service could not process that prompt or image. Try a shorter, "
            "clearer prompt and a smaller PNG or JPEG image.",
            status_code=400,
            blocked=True,
        )
    if status_code in (401, 403):
        return ImageGenerationError(
            "Image generation is not configured correctly right now. Please contact support.",
        )
    if status_code == 404:
        return ImageGenerationError(
            "This image model is not available on the connected account right now.",
        )
    return ImageGenerationError(
        "The image service is temporarily unavailable. Please try again in a moment.",
    )


def _size_for_ratio(width_units, height_units):
    """Snap any requested ratio to the nearest size FLUX actually accepts."""
    if width_units <= 0 or height_units <= 0:
        return None
    target = width_units / height_units
    # Ignore absurd ratios rather than generating a 20:1 sliver — they are far
    # more likely to be a misread number than a real request.
    if not 0.2 <= target <= 5:
        return None
    return min(
        _ASPECT_SIZES.values(),
        key=lambda size: abs((size[0] / size[1]) - target),
    )


def resolve_dimensions(prompt):
    """Work out the width/height a prompt is asking for.

    Falls back to the square default whenever nothing is stated, so an ordinary
    "draw a cat" behaves exactly as it always has. Pure string work — no model
    call, no network — so this costs nothing measurable on the request path.
    """
    text = prompt or ""

    match = _PIXEL_SIZE_RE.search(text)
    if match:
        size = _size_for_ratio(int(match.group(1)), int(match.group(2)))
        if size:
            return size

    # A ratio introduced by an explicit size word is taken at face value; a
    # bare one has to look like a real aspect ratio first (see the allowlist).
    for found in _SIZED_RATIO_RE.finditer(text):
        size = _size_for_ratio(int(found.group(1)), int(found.group(2)))
        if size:
            return size
    for found in _RATIO_SIZED_RE.finditer(text):
        size = _size_for_ratio(int(found.group(1)), int(found.group(2)))
        if size:
            return size
    for found in _RATIO_RE.finditer(text):
        pair = (int(found.group(1)), int(found.group(2)))
        if pair in _BARE_RATIO_ALLOWLIST:
            size = _size_for_ratio(*pair)
            if size:
                return size

    for pattern, ratio in _ORIENTATION_CUES:
        if pattern.search(text):
            return _ASPECT_SIZES[ratio]

    return DEFAULT_SIZE


def _shorten_prompt(prompt, limit):
    """Cut ``prompt`` down to ``limit`` chars, preferring a sentence/word
    boundary so the trimmed prompt still reads as a complete thought."""
    if len(prompt) <= limit:
        return prompt
    window = prompt[:limit]
    for boundary in (". ", "! ", "? "):
        cut = window.rfind(boundary)
        if cut != -1 and cut >= limit * 0.4:
            return window[: cut + 1].strip()
    cut = window.rfind(" ")
    return (window[:cut] if cut >= limit * 0.4 else window).strip()


# Phrases that plausibly trip a content filter not because the *picture*
# they describe is unsafe, but because the prompt reads like it's staging a
# security-bypass/data-leak scene — quoted "SYSTEM PROMPT" / "ACCESS DENIED"
# style warning text, asked to be rendered literally. Stripped out before a
# retry rather than guessed at case by case.
_PROMPT_BLOCK_TRIGGER_RE = re.compile(
    r"\b(?:system prompt|hidden instructions?|private data|model identity|"
    r"internal configuration|jailbreak|access denied|extract(?:ing)? the ai'?s?)\b",
    re.IGNORECASE,
)
_QUOTED_TEXT_RE = re.compile(r'["“][^"”]{1,80}["”]')




def _api_key(*, editing=False):
    # Editing can use a separately entitled NVIDIA account/key while normal
    # prompt-to-image generation keeps its existing credential.
    setting_name = "NVIDIA_FLUX_EDIT_API_KEY" if editing else "NVIDIA_FLUX_API_KEY"
    return _get_key(setting_name).strip()


_IMAGE_DATA_URI_RE = re.compile(
    r"^data:image/(?:png|jpe?g|webp);base64,(?P<data>[A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)


def _normalize_source_image(data_uri):
    """Decode, orient, resize and re-encode a browser image for FLUX."""
    match = _IMAGE_DATA_URI_RE.fullmatch(data_uri or "")
    if not match:
        raise ImageGenerationError(
            "The attached image is not a valid PNG, JPEG, or WebP image.",
            status_code=400,
        )
    try:
        raw = base64.b64decode(match.group("data"), validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageGenerationError(
            "The attached image could not be decoded. Try uploading it again as PNG or JPEG.",
            status_code=400,
        ) from exc

    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        # JPEG has no alpha channel. Flatten transparent pixels onto white so
        # they do not unexpectedly become black in the model input.
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        image = background.convert("RGB")

    encoded = b""
    for quality in (88, 78, 68, 58):
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = output.getvalue()
        if len(encoded) <= 600_000:
            break
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _decode_artifact(payload):
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    artifact = artifacts[0] if artifacts and isinstance(artifacts[0], dict) else None
    finish_reason = artifact.get("finishReason") if artifact else None
    if finish_reason == "CONTENT_FILTERED":
        raise ImageGenerationError(
            "That image request was blocked by the content filter. Try a different prompt or image.",
            status_code=400,
            blocked=True,
        )
    if finish_reason == "ERROR":
        raise ImageGenerationError("The image service could not generate that image. Please try again.")
    encoded = artifact.get("base64") if artifact else None
    if not isinstance(encoded, str) or not encoded:
        raise ImageGenerationError("The image service returned no image. Please try again.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageGenerationError("The image service returned an unreadable image. Please try again.")
    return _validate_image_bytes(content)


def _validate_image_bytes(content, *, service_name="The image service"):
    """Detect the format, fully decode, and reject a blank placeholder frame.

    Shared by every image backend: a matching file
    signature alone is not proof of a usable image — a truncated payload
    used to be stored and returned as a broken image (report #51 also saw
    all-white/all-black placeholder frames presented as a real generation).
    """
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = "png"
    elif content.startswith(b"\xff\xd8\xff"):
        extension = "jpg"
    elif content.startswith((b"RIFF",)) and content[8:12] == b"WEBP":
        extension = "webp"
    else:
        raise ImageGenerationError(f"{service_name} returned an unsupported image format. Please try again.")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            opened.load()
            sample = opened.convert("RGB")
            sample.thumbnail((64, 64), Image.Resampling.BILINEAR)
            extrema = sample.getextrema()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageGenerationError(
            "The image service returned a damaged image. Please try again."
        ) from exc
    nearly_white = all(low >= 250 and high >= 250 for low, high in extrema)
    nearly_black = all(low <= 5 and high <= 5 for low, high in extrema)
    if nearly_white or nearly_black:
        raise ImageGenerationError(
            "The image service returned a blank image. Please try again."
        )
    return GeneratedImage(content=content, extension=extension)


def _generate_qwen_edit(prompt, source_image):
    if not source_image:
        raise ImageGenerationError('Attach an image and describe the changes you want.', status_code=400)
    url = _get_key('QWEN_IMAGE_EDIT_API_URL').strip()
    if not url:
        raise ImageGenerationError('Qwen Image Edit is not connected yet. An image-editing server must be configured before uploads can be edited.')
    key = _get_key('QWEN_IMAGE_EDIT_ENDPOINT_KEY').strip()
    try:
        response = requests.post(
            url,
            headers={'Accept': 'application/json', **({'Authorization': f'Bearer {key}'} if key else {})},
            json={'prompt': prompt, 'image': _normalize_source_image(source_image), 'seed': 0},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ImageGenerationError('Could not reach the image-editing server. Please try again.') from exc
    if response.status_code != 200:
        raise ImageGenerationError('The image-editing server could not complete this edit. Please try again later.')
    try:
        return _decode_artifact(response.json())
    except ValueError as exc:
        raise ImageGenerationError('The image-editing server returned an invalid response.') from exc


def generate_image(prompt, source_image=None, *, model_key=None):
    """Generate an image, or edit ``source_image`` when one is supplied.

    ``source_image`` is the browser-provided PNG/JPEG data URI. It is decoded
    and normalized before being placed in FLUX's reference-image array.

    There is no fallback: if the selected backend fails or blocks the prompt,
    the error is raised as is rather than retried on another key or provider.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ImageGenerationError(
            "Describe the image you want to generate or how you want the attached image changed.",
            status_code=400,
        )
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ImageGenerationError("That image prompt is too long.", status_code=400)

    # One attempt on the one configured service: a failure is reported, not
    # silently retried on another key or provider.
    return _dispatch_generate(prompt, source_image, model_key)


def _dispatch_generate(prompt, source_image, model_key):
    if model_key == 'qwen-image-edit':
        return _generate_qwen_edit(prompt, source_image)

    editing = bool(source_image)
    kontext = model_key == 'flux-kontext-dev'
    if kontext and not editing:
        raise ImageGenerationError('Attach an image and describe the changes you want.', status_code=400)
    edit_url = _get_key('FLUX_EDIT_API_URL').strip() if editing else ''
    # NVIDIA's hosted FLUX.2 Klein preview does not accept arbitrary uploads.
    # Its ``image`` field only accepts one of four NVIDIA-owned example IDs
    # (data:image/png;example_id,0..3), so sending a browser image as base64
    # can never work there.  Real uploads must go to an upload-capable NIM
    # configured through FLUX_EDIT_API_URL.  Signal the caller immediately so
    # it can use the existing describe-and-regenerate fallback without making
    # a guaranteed-to-fail, potentially billable hosted request first.
    if editing and not edit_url and not kontext:
        raise ImageGenerationError(
            'Uploaded-image editing needs an upload-capable image-editing server.',
            status_code=503,
            editing_unavailable=True,
        )
    # A private deployment has its own optional credential. Never forward the
    # hosted NVIDIA credential to a separately configured server.
    key = _get_key('FLUX_EDIT_API_KEY').strip() if edit_url else _api_key(editing=editing)
    if kontext:
        edit_url = ''
        key = _get_key('NVIDIA_FLUX_KONTEXT_API_KEY').strip()
    if not key and not edit_url:
        setting_name = 'NVIDIA_FLUX_KONTEXT_API_KEY' if kontext else ("NVIDIA_FLUX_EDIT_API_KEY" if editing else "NVIDIA_FLUX_API_KEY")
        raise ImageGenerationError(
            f"Image generation is not configured yet. Set {setting_name} on the server.",
        )

    width, height = resolve_dimensions(prompt)
    # Dimensions are read from the full prompt (a size/ratio cue could sit
    # anywhere), but the text actually sent to FLUX is shortened — see
    # FLUX_SAFE_PROMPT_CHARS above. Kontext edits keep the full prompt: it's
    # already proven to accept longer text in practice.
    flux_prompt = prompt if kontext else _shorten_prompt(prompt, FLUX_SAFE_PROMPT_CHARS)
    body = {
        "prompt": flux_prompt,
        "width": width,
        "height": height,
        "steps": 4,
        "samples": 1,
        "seed": 0,
    }
    if editing:
        # FLUX.2 supports multiple references; NVIDIA's current hosted
        # request template therefore expects an array even for one image.
        body["image"] = [_normalize_source_image(source_image)]
    if kontext:
        body['image'] = body['image'][0]
        body['steps'] = 30
        body.pop('width')
        body.pop('height')
        body['aspect_ratio'] = 'match_input_image'

    try:
        response = requests.post(
            ('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev'
             if kontext else edit_url or FLUX_API_URL),
            headers={
                **({"Authorization": f"Bearer {key}"} if key else {}),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ImageGenerationError(
            "Could not reach the image service. Please try again."
        ) from exc

    if response.status_code != 200:
        raise _safe_error(response)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ImageGenerationError("The image service returned an invalid response. Please try again.") from exc
    return _decode_artifact(payload)


def test_connection():
    """Generate one tiny test picture with the FLUX key currently in effect
    (and its backup, if set), so the dashboard can confirm the key works.
    Returns (ok, message)."""
    try:
        image = _dispatch_generate('a small red circle on a plain white background', None, None)
    except ImageGenerationError as exc:
        return False, str(exc)[:250]
    except Exception as exc:
        return False, f'{exc.__class__.__name__}: {str(exc)[:200]}'
    return True, f'Working — generated a test image ({len(image.content) // 1024} KB).'
