"""Chat models added on the dashboard's API Settings page ("Add a new model").

A model is added only after a set of live checks shows it really works the way
the chat needs it to: the key looks right for the API type, the model name
exists, it is a chat model, it answers and streams, and whether it accepts the
"no visible thinking" switch and can read images. Pasted values are cleaned
up first (a build.nvidia.com page link becomes its model name, "Bearer " and
quotes are removed from a key, and so on).

Added models are kept in models.CustomAIModel and served as ai_chat.MODELS
entries keyed ``custom-<pk>``; sync() keeps MODELS in step with the table.
"""
import base64
import difflib
import io
import logging
import re
import time

import requests
from django.core.cache import cache
from django.utils import timezone

from myapp import ai_chat

logger = logging.getLogger(__name__)

PREFIX = 'custom-'
NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'
CHECK_TIMEOUT = 45.0
_VERSION_KEY = 'custom_models_version'
_RESYNC_SECONDS = 20

# Words in a model name that mean it is not a chat model at all.
_NOT_CHAT = [
    (re.compile(r'embed|retriever|nemoretriever', re.I), 'an embedding (search) model'),
    (re.compile(r'rerank', re.I), 'a reranking model'),
    (re.compile(r'reward', re.I), 'a reward-scoring model'),
    (re.compile(r'safety|guard', re.I), 'a safety classifier'),
    (re.compile(r'nemotron-parse|ocdrnet|paddleocr', re.I), 'a document-reading model'),
    (re.compile(r'flux|stable-diffusion|sdxl|bria|cosmos|edify|trellis', re.I), 'an image/video generation model'),
    (re.compile(r'parakeet|canary|whisper|riva|fastpitch|magpie|audio2face|tts|asr', re.I), 'a speech model'),
]
_VISION_HINT = re.compile(r'vision|[-_]vl\b|[-_]vl[-_]|vlm|omni|pixtral|gemma-3|llava|kosmos|neva|paligemma', re.I)

_last_sync = {'version': None, 'at': 0.0}


def is_custom(model_key):
    return str(model_key or '').startswith(PREFIX)


# ----------------------------------------------------------- cleaning input

def clean_model_id(raw):
    """The model name from whatever was pasted: the name itself, a
    build.nvidia.com page link, or a docs link. Quotes/spaces are dropped."""
    value = (raw or '').strip().strip('"\'` ').strip()
    match = re.search(r'build\.nvidia\.com/([\w.-]+)/([\w.-]+)', value)
    if match:
        return f'{match.group(1)}/{match.group(2)}'
    match = re.search(r'docs\.api\.nvidia\.com/nim/reference/([\w.-]+?)-([\w.-]+?)(?:-infer)?/?$', value)
    if match:
        return f'{match.group(1)}/{match.group(2)}'
    value = re.sub(r'^https?://\S+?/models?/', '', value)
    return re.sub(r'\s+', '', value)


def clean_key(raw):
    value = (raw or '').strip().strip('"\'` ').strip()
    value = re.sub(r'^(?:authorization:\s*)?bearer\s+', '', value, flags=re.I)
    value = re.sub(r'^\w*API_KEY\s*=\s*', '', value)
    return re.sub(r'\s+', '', value.strip('"\'` '))


def clean_base_url(raw):
    value = (raw or '').strip().strip('"\'` ').rstrip('/')
    if value and not re.match(r'^https?://', value, re.I):
        value = 'https://' + value
    return re.sub(r'/(?:chat/)?completions$', '', value).rstrip('/')


_ACRONYMS = {'gpt', 'oss', 'ai', 'llm', 'vl', 'qa', 'moe', 'r1', 'it'}


def default_name(model_id):
    """'nvidia/nemotron-3-super-120b-a12b' -> 'Nemotron 3 Super 120B A12B'."""
    words = re.split(r'[-_\s]+', model_id.split('/')[-1])
    out = []
    for word in words:
        if word.lower() in _ACRONYMS:
            out.append(word.upper())
        elif re.fullmatch(r'\d+(?:\.\d+)?[bm]|a\d+b|v\d+(?:\.\d+)?', word, re.I):
            out.append(word.upper() if not word.lower().startswith('v') else word.lower())
        else:
            out.append(word[:1].upper() + word[1:])
    return ' '.join(out)[:60]


# ------------------------------------------------------------------- checks

def _explain_unlisted(model_id):
    """Why a model on build.nvidia.com is not on the chat API, from its
    catalog entry ('' when the catalog does not know it either)."""
    try:
        from myapp import api_checker
        models, _info = api_checker.build_models()
    except Exception:
        return ''
    wanted = api_checker._match_key(model_id.split('/')[-1])
    entry = next((m for m in models if api_checker._match_key(m['id'].split('/')[-1]) == wanted), None)
    if not entry or entry['on_api']:
        return ''
    tags = {t.lower() for t in entry['tags']}
    what = {
        'image': 'an image generation / editing model', 'speech': 'a speech model',
        'retrieval': 'a search (embedding) model', 'safety': 'a safety checker',
        'science': 'a biology / science model',
    }.get(entry['kind'], '')
    parts = [f"{entry['id']} is on build.nvidia.com" + (f", but it is {what}, not a chat model" if what else '') + '.']
    if 'free endpoint' not in tags and 'download available' in tags and 'partner endpoint' not in tags:
        parts.append('NVIDIA only offers it as a download to run on your own GPU server (the "Downloadable" badge on its page) '
                     '— there is no hosted API for it, so no API key can use it.')
    elif 'partner endpoint' in tags and 'free endpoint' not in tags:
        parts.append('NVIDIA only offers it through partner clouds (such as Together AI), not with an NVIDIA API key.')
    else:
        parts.append('NVIDIA serves it on a separate address made for that kind of model, not the chat API, so it cannot be added to the chat.')
    parts.append('Only models marked "Free Endpoint" with chat abilities can be added — API Checker in the sidebar lists them.')
    return ' '.join(parts)


def _item(label, status, detail):
    return {'label': label, 'status': status, 'detail': detail}


def _catalog(base_url, api_key):
    """The provider's list of model names, or None if it cannot be read."""
    try:
        response = requests.get(
            f'{base_url}/models', timeout=15,
            headers={'Authorization': f'Bearer {api_key}'} if api_key else {},
        )
        if response.status_code != 200:
            return None
        return [m.get('id') for m in response.json().get('data', []) if m.get('id')]
    except Exception:
        return None


def _explain_error(exc, model_id):
    status = getattr(exc, 'status_code', None)
    text = str(exc)
    lowered = text.lower()
    if status in (401, 403) or 'invalid api key' in lowered or 'unauthorized' in lowered:
        return 'fail', 'The provider rejected this API key (wrong key, or it has no access to this model).'
    if status == 404:
        return 'fail', f'{model_id} is not available to this key, or it is not a chat model.'
    if status == 429:
        return 'warn', 'The key works, but the provider is rate-limiting it right now. Try the test again in a minute.'
    if 'timeout' in exc.__class__.__name__.lower() or 'timed out' in lowered:
        return 'fail', f'No answer within {int(CHECK_TIMEOUT)} seconds — the model is busy or queued. Try again shortly.'
    if 'connection' in exc.__class__.__name__.lower():
        return 'fail', 'Could not reach the API address. Check the link and your internet connection.'
    return 'fail', f'{exc.__class__.__name__}: {text[:200]}'


def _stream_reply(client, model_id, messages, extra_body):
    """Stream one short reply. Returns (text, seconds_to_first_word)."""
    started = time.monotonic()
    first = None
    parts = []
    kwargs = dict(model=model_id, messages=messages, max_tokens=200, temperature=0.2,
                  stream=True, timeout=CHECK_TIMEOUT)
    if extra_body:
        kwargs['extra_body'] = extra_body
    stream = client.chat.completions.create(**kwargs)
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, 'content', None)
            if content:
                if first is None:
                    first = time.monotonic() - started
                parts.append(content)
    finally:
        close = getattr(stream, 'close', None)
        if close:
            close()
    return ''.join(parts), first


def _test_image_data_url():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (96, 96), (220, 20, 30)).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def run_checks(api_type, base_url, model_id, api_key):
    """Every check, in order. Returns (ok, report, found) where found holds
    thinking_control and vision. ok means the model can be used for chat."""
    report = []
    found = {'thinking_control': False, 'vision': False}
    nvidia = api_type == 'nvidia'
    base = NVIDIA_BASE_URL if nvidia else base_url

    # 1. The pasted details.
    if not model_id:
        return False, [_item('Model name', 'fail', 'Enter the model name, e.g. nvidia/nemotron-3-super-120b-a12b.')], found
    if not api_key:
        return False, [_item('API key', 'fail', 'Paste the API key for this model.')], found
    if not nvidia and not re.match(r'^https?://[^\s/]+', base or ''):
        return False, [_item('API address', 'fail', 'Enter the API address, e.g. https://api.example.com/v1.')], found
    if nvidia and not api_key.startswith('nvapi-'):
        return False, [_item('API key', 'fail', 'NVIDIA keys start with "nvapi-". Copy the key again from build.nvidia.com → "Get API Key".')], found
    report.append(_item('API key', 'ok', 'The key format looks right (starts with nvapi-); it is tried for real in the test message below.' if nvidia else 'Key entered; it is tried for real in the test message below.'))

    for pattern, what in _NOT_CHAT:
        if pattern.search(model_id):
            report.append(_item('Chat model', 'fail', f'{model_id} looks like {what}, not a chat model. Only chat models can be added to the model picker.'))
            return False, report, found

    # 2. The name exists in the provider's catalog.
    catalog = _catalog(base, api_key)
    if catalog is not None:
        if model_id in catalog:
            report.append(_item('Model name', 'ok', f'{model_id} is in the {"NVIDIA" if nvidia else "provider"} catalog.'))
        else:
            exact = next((c for c in catalog if c.lower() == model_id.lower()), None)
            # "qwen-image-edit" typed without its "qwen/" publisher part.
            prefixed = None if '/' in model_id else next(
                (c for c in catalog if c.lower().split('/')[-1] == model_id.lower()), None)
            if exact or prefixed:
                fixed = exact or prefixed
                report.append(_item('Model name', 'ok', f'Corrected the name to {fixed}.'))
                model_id = fixed
                found['model_id'] = fixed
            elif nvidia and _explain_unlisted(model_id):
                report.append(_item('Model name', 'fail', _explain_unlisted(model_id)))
                return False, report, found
            else:
                close = difflib.get_close_matches(model_id, catalog, n=3, cutoff=0.6)
                hint = f' Did you mean {" or ".join(close)}?' if close else ' Copy the name from the model\'s page on build.nvidia.com.' if nvidia else ''
                report.append(_item('Model name', 'fail', f'{model_id} is not in the provider\'s model list.{hint}'))
                return False, report, found
    else:
        report.append(_item('Model name', 'warn', 'Could not read the provider\'s model list, so the name is checked by the test message below instead.'))

    # 3. It answers, streaming, with or without the thinking switch.
    client = ai_chat._client_for_key(api_key, base)
    messages = [{'role': 'user', 'content': 'Reply with the single word OK.'}]
    thinking_body = {'chat_template_kwargs': {'enable_thinking': False, 'force_nonempty_content': True}} if nvidia else None
    reply = None
    first = None
    try:
        if thinking_body:
            try:
                reply, first = _stream_reply(client, model_id, messages, thinking_body)
                found['thinking_control'] = True
            except Exception as exc:
                # Only a complaint about the switch itself means "try without it".
                message = str(exc).lower()
                if not (getattr(exc, 'status_code', None) in (400, 422)
                        and any(word in message for word in ('template', 'kwargs', 'extra'))):
                    raise
        if reply is None:
            reply, first = _stream_reply(client, model_id, messages, None)
    except Exception as exc:
        status, detail = _explain_error(exc, model_id)
        report.append(_item('Test message', status, detail))
        # A rate-limited key is a working key; it can still be added.
        return status == 'warn', report, found

    shown = ai_chat.strip_think_tags(reply).strip()
    if not shown:
        report.append(_item('Test message', 'fail', 'The model answered with no text. It may not be a chat model, or it only returns its hidden thinking.'))
        return False, report, found
    report.append(_item('Test message', 'ok', f'Replied "{shown[:60]}" — first words after {first:.1f} s.'))
    report.append(_item('Live streaming', 'ok', 'Replies arrive word by word, like the rest of the chat.'))
    if found['thinking_control']:
        report.append(_item('Thinking switch', 'ok', 'Accepts the switch that hides its thinking, so replies stay clean.'))
    elif nvidia:
        report.append(_item('Thinking switch', 'ok', 'Not needed for this model.'))
    if '<think>' in (reply or ''):
        report.append(_item('Hidden thinking', 'warn', 'This model writes out its thinking; it is removed from replies automatically.'))

    # 4. Images, when the name suggests the model can read them.
    if _VISION_HINT.search(model_id):
        image_messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What single colour fills this picture? Answer with one word.'},
            {'type': 'image_url', 'image_url': {'url': _test_image_data_url()}},
        ]}]
        try:
            seen, _first = _stream_reply(client, model_id, image_messages,
                                         thinking_body if found['thinking_control'] else None)
            if 'red' in ai_chat.strip_think_tags(seen).lower():
                found['vision'] = True
                report.append(_item('Reads images', 'ok', 'Correctly described a test picture, so photos can be sent to it.'))
            else:
                report.append(_item('Reads images', 'warn', 'Answered, but did not describe the test picture correctly; photos will go to the built-in vision model instead.'))
        except Exception:
            report.append(_item('Reads images', 'warn', 'Did not accept a test picture; photos will go to the built-in vision model instead.'))
    else:
        report.append(_item('Reads images', 'ok', 'Text only — attached photos go to the built-in vision model.'))
    return True, report, found


# --------------------------------------------------------------- the picker

def _config(row):
    return {
        'id': row.model_id,
        'label': row.display_name,
        'description': row.description or f'Added model: {row.model_id}',
        'reasoning': row.thinking_control,
        'vision': row.vision,
        'api_key_setting': row.key_setting,
        'base_url': NVIDIA_BASE_URL if row.api_type == row.API_NVIDIA else row.base_url,
        'timeout': ai_chat.STREAM_TIMEOUT_LONG,
        'custom': True,
    }


def mark_changed():
    cache.set(_VERSION_KEY, time.time(), None)
    sync(force=True)


def sync(force=False):
    """Put every added model into ai_chat.MODELS (and drop deleted ones).
    Cheap when nothing changed: re-reads the table only after a change in
    this process or every few seconds, so other workers catch up too."""
    version = cache.get(_VERSION_KEY)
    now = time.monotonic()
    if not force and version == _last_sync['version'] and now - _last_sync['at'] < _RESYNC_SECONDS:
        return
    try:
        from myapp.models import CustomAIModel
        rows = list(CustomAIModel.objects.all())
    except Exception:
        return
    wanted = {row.model_key: _config(row) for row in rows}
    for key in [k for k in ai_chat.MODELS if is_custom(k) and k not in wanted]:
        del ai_chat.MODELS[key]
    ai_chat.MODELS.update(wanted)
    _last_sync.update(version=version, at=now)


def save_report(row, report):
    row.check_report = report
    row.last_checked_at = timezone.now()
