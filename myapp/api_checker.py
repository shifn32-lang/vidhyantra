"""Dashboard → API Checker: what an NVIDIA API key can do.

Two NVIDIA sources are merged into one model list:

* the chat API's own model list (integrate.api.nvidia.com/v1/models) — the
  models that can be called with a key, but names only, and the same list for
  every key;
* the build.nvidia.com catalog search (api.ngc.nvidia.com) — descriptions,
  capability tags, publisher, popularity and release date. It is public but
  not officially documented, so every field is optional and capabilities fall
  back to clues in the model name when the catalog has nothing.

Whether a key works, and whether it can use a given model, is only known by
sending a tiny message — see check_key() and test_model(). Keys are used for
the check and saved with its results in the checker's History
(models.APICheckRun / APICheckResult), so a key can be re-used later.
"""
import json
import logging
import re
import time

import requests
from django.core.cache import cache

from myapp import ai_chat, custom_models

logger = logging.getLogger(__name__)

MODELS_URL = 'https://integrate.api.nvidia.com/v1/models'
CATALOG_URL = 'https://api.ngc.nvidia.com/v2/search/catalog/resources/ENDPOINT'
CACHE_KEY = 'api_checker_catalog_v1'
CACHE_SECONDS = 60 * 60
TEST_TIMEOUT = 30.0
KEY_TEST_MODEL = 'meta/llama-3.1-8b-instruct'

# Capability -> (label shown on the page, catalog tags that mean it, name clue)
CAPABILITIES = [
    ('chat', 'Chat', {'chat', 'text-to-text', 'large language models', 'instruction following', 'llm'}, None),
    ('reasoning', 'Reasoning', {'reasoning'}, re.compile(r'reason|thinking|qwq|-r1\b|r1-|deepseek-r1|magistral', re.I)),
    ('vision', 'Reads images', {'image-to-text', 'vision language model', 'multimodal', 'visual question answering', 'vlm'}, custom_models._VISION_HINT),
    ('code', 'Coding', {'code generation', 'coding', 'code'}, re.compile(r'code|coder|starcoder|codestral|devstral', re.I)),
    ('tools', 'Tool calling', {'tool use', 'tool calling', 'function calling', 'agentic'}, None),
    ('long_context', 'Long context', {'long context'}, None),
    ('image_generation', 'Makes images', {'image generation', 'text-to-image'}, re.compile(r'flux|stable-diffusion|sdxl|bria', re.I)),
    ('speech', 'Speech', {'asr', 'speech-to-text', 'tts', 'text-to-speech', 'automatic speech recognition'}, re.compile(r'parakeet|canary|whisper|riva|fastpitch|magpie', re.I)),
    ('retrieval', 'Search / embeddings', {'nemo retriever', 'embedding', 'retrieval', 'reranking'}, re.compile(r'embed|rerank|retriever', re.I)),
    ('safety', 'Safety', {'content safety', 'safety', 'guardrails'}, re.compile(r'safety|guard', re.I)),
    ('science', 'Biology / science', {'drug discovery', 'biology', 'bionemo', 'protein folding', 'chemistry', 'dna generation'}, None),
    ('free', 'Free endpoint', {'free endpoint'}, None),
]
CAPABILITY_LABELS = {key: label for key, label, _tags, _clue in CAPABILITIES}


def _catalog_page(page):
    query = {'query': '*', 'page': page, 'pageSize': 100}
    response = requests.get(CATALOG_URL, params={'q': json.dumps(query)}, timeout=25)
    response.raise_for_status()
    data = response.json()
    items = [item for group in data.get('results', []) if group.get('groupValue') == 'ENDPOINT'
             for item in group.get('resources', [])]
    return items, int(data.get('resultPageTotal') or 1)


def _fetch_catalog():
    items, pages = _catalog_page(0)
    for page in range(1, min(pages, 10)):
        more, _pages = _catalog_page(page)
        items.extend(more)
    return items


def _fetch_api_models():
    response = requests.get(MODELS_URL, timeout=20)
    response.raise_for_status()
    return [m['id'] for m in response.json().get('data', []) if m.get('id')]


def _labels(item, key):
    for label in item.get('labels') or []:
        if label.get('key') == key:
            return [str(v) for v in label.get('values') or []]
    return []


def _attr(item, key):
    for attr in item.get('attributes') or []:
        if attr.get('key') == key:
            return attr.get('value')
    return None


def _capabilities(model_id, tags, on_api):
    lowered = {t.lower() for t in tags}
    found = []
    for key, _label, tag_set, clue in CAPABILITIES:
        if lowered & tag_set or (clue and clue.search(model_id)):
            found.append(key)
    not_chat = (any(pattern.search(model_id) for pattern, _what in custom_models._NOT_CHAT)
                or re.search(r'detector|classifier', model_id, re.I))
    if on_api and not not_chat and 'chat' not in found and not ({'science', 'speech', 'image_generation'} & set(found)):
        found.insert(0, 'chat')
    if not_chat and 'chat' in found:
        found.remove('chat')
    return found


def _match_key(name):
    """The catalog writes some names differently ("llama-3_1-…" for "llama-3.1-…")."""
    return re.sub(r'[._]', '-', name.lower())


def _kind(caps):
    for key, kind in (('chat', 'chat'), ('retrieval', 'retrieval'), ('image_generation', 'image'),
                      ('speech', 'speech'), ('safety', 'safety'), ('science', 'science')):
        if key in caps:
            return kind
    return 'other'


def build_models():
    """The merged model list (cached for an hour). Returns (models, info)."""
    cached = cache.get(CACHE_KEY)
    if cached:
        return cached
    info = {'catalog_ok': True, 'api_ok': True, 'fetched_at': time.time()}
    try:
        api_ids = _fetch_api_models()
    except Exception:
        logger.warning('API Checker could not read the NVIDIA model list', exc_info=True)
        api_ids, info['api_ok'] = [], False
    try:
        catalog = _fetch_catalog()
    except Exception:
        logger.warning('API Checker could not read the NVIDIA catalog', exc_info=True)
        catalog, info['catalog_ok'] = [], False

    # The catalog lists some models twice; keep the more popular entry.
    by_name = {}
    for item in catalog:
        name = _match_key(item.get('name') or '')
        if not name:
            continue
        current = by_name.get(name)
        if current is None or (item.get('weightPopular') or 0) > (current.get('weightPopular') or 0):
            by_name[name] = item

    models = []
    seen = set()

    def add(model_id, item, on_api):
        tags = _labels(item, 'general') if item else []
        publisher = (_labels(item, 'publisher') or [model_id.split('/')[0] if '/' in model_id else ''])[0] if item else model_id.split('/')[0]
        caps = _capabilities(model_id, tags, on_api)
        calls = _attr(item, 'lastMonthApiInvocationCount') if item else None
        models.append({
            'id': model_id,
            'name': (item.get('displayName') or item.get('name')) if item else model_id.split('/')[-1],
            'publisher': publisher,
            'description': (item.get('description') or '').strip() if item else '',
            'tags': sorted({t for t in tags if t.lower() not in ('chat',)}, key=str.lower)[:12],
            'caps': caps,
            'kind': _kind(caps),
            'on_api': on_api,
            'in_catalog': bool(item),
            'guessed': not item,
            'calls_last_month': int(calls) if calls and str(calls).isdigit() else 0,
            'released': (item.get('dateCreated') or '')[:10] if item else '',
            'logo': _attr(item, 'logo') or '' if item else '',
            'preview': (_attr(item, 'PREVIEW') == 'true') if item else False,
            'testable': on_api and 'chat' in caps,
        })

    for model_id in api_ids:
        item = by_name.get(_match_key(model_id.split('/')[-1]))
        add(model_id, item, True)
        seen.add(_match_key(model_id.split('/')[-1]))
    for name, item in by_name.items():
        if name in seen:
            continue
        publisher = (_labels(item, 'publisher') or ['nvidia'])[0]
        add(f'{publisher}/{item.get("name")}', item, False)

    models.sort(key=lambda m: (-m['calls_last_month'], m['id']))
    result = (models, info)
    if api_ids or catalog:
        cache.set(CACHE_KEY, result, CACHE_SECONDS)
    return result


def summary(models):
    counts = {key: 0 for key, *_rest in CAPABILITIES}
    for m in models:
        for cap in m['caps']:
            counts[cap] = counts.get(cap, 0) + 1
    return {
        'total': len(models),
        'on_api': sum(1 for m in models if m['on_api']),
        'testable': sum(1 for m in models if m['testable']),
        'catalog_only': sum(1 for m in models if not m['on_api']),
        'caps': [{'key': key, 'label': CAPABILITY_LABELS[key], 'count': counts.get(key, 0)}
                 for key, *_rest in CAPABILITIES if counts.get(key)],
    }


# ------------------------------------------------------------- live tests

def _result(status, detail, seconds=None, reply=''):
    return {'status': status, 'detail': detail, 'seconds': round(seconds, 2) if seconds is not None else None, 'reply': reply}


def test_model(api_key, model_id):
    """One tiny streamed message to one model with this key."""
    api_key = custom_models.clean_key(api_key)
    if not api_key.startswith('nvapi-'):
        return _result('bad_key', 'NVIDIA keys start with "nvapi-".')
    client = ai_chat._client_for_key(api_key, custom_models.NVIDIA_BASE_URL)
    messages = [{'role': 'user', 'content': 'Reply with the single word OK.'}]
    started = time.monotonic()
    thinking_body = {'chat_template_kwargs': {'enable_thinking': False, 'force_nonempty_content': True}}
    try:
        try:
            reply, first = _reply(client, model_id, messages, thinking_body)
        except Exception as exc:
            message = str(exc).lower()
            if not (getattr(exc, 'status_code', None) in (400, 422)
                    and any(word in message for word in ('template', 'kwargs', 'extra'))):
                raise
            reply, first = _reply(client, model_id, messages, None)
    except Exception as exc:
        status = getattr(exc, 'status_code', None)
        lowered = str(exc).lower()
        if status in (401, 403) or 'unauthorized' in lowered or 'invalid api key' in lowered:
            return _result('no_access', 'This key is not allowed to use this model (or the key is wrong).')
        if status == 404:
            return _result('retired', 'NVIDIA still lists it but no longer serves it on the API — this is not about your key.')
        if re.search(r'overloaded|resourceexhausted|request limit|service unavailable|capacity', lowered) or status == 503:
            return _result('busy', 'NVIDIA is overloaded for this model right now. Try again later.')
        if status == 429:
            return _result('busy', 'Rate limit reached — wait a minute and test again.')
        if 'timeout' in exc.__class__.__name__.lower() or 'timed out' in lowered:
            return _result('busy', f'No answer within {int(TEST_TIMEOUT)} seconds — the model is busy or queued.')
        if status and 500 <= status < 600:
            return _result('busy', 'NVIDIA had a server problem with this model. Try again later.')
        return _result('error', f'{exc.__class__.__name__}: {str(exc)[:160]}')
    text = ai_chat.strip_think_tags(reply).strip()
    if not text:
        return _result('error', 'Replied with no text in the short test (it may have spent it on hidden thinking). Test it again.',
                       time.monotonic() - started)
    return _result('works', f'Replied "{text[:40]}"', first if first is not None else time.monotonic() - started, text[:60])


def _reply(client, model_id, messages, extra_body):
    started = time.monotonic()
    first = None
    parts = []
    kwargs = dict(model=model_id, messages=messages, max_tokens=200, temperature=0.2, stream=True, timeout=TEST_TIMEOUT)
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


def check_key(api_key, models=None):
    """Is this a working NVIDIA key? One tiny message to a small, fast model."""
    api_key = custom_models.clean_key(api_key)
    if not api_key:
        return _result('bad_key', 'Paste an NVIDIA API key first.')
    if not api_key.startswith('nvapi-'):
        return _result('bad_key', 'NVIDIA keys start with "nvapi-". Copy the key again from build.nvidia.com → "Get API Key".')
    model_id = KEY_TEST_MODEL
    if models and not any(m['id'] == model_id for m in models):
        model_id = next((m['id'] for m in models if m['testable']), ai_chat.NVIDIA_CHAT_MODEL)
    result = test_model(api_key, model_id)
    if result['status'] == 'works':
        result['detail'] = f'The key works — {model_id} answered in {result["seconds"]} s.'
    elif result['status'] == 'no_access':
        result['status'] = 'bad_key'
        result['detail'] = 'NVIDIA rejected this key. Check that it was copied completely.'
    elif result['status'] == 'busy':
        result['detail'] = 'The key was not rejected, but NVIDIA is busy or rate-limiting it right now. ' + result['detail']
    result['model'] = model_id
    return result



# ------------------------------------------------------------------ views

def _json_body(request):
    try:
        data = json.loads(request.body or '{}')
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _staff_json(view):
    """Staff-only JSON endpoint (POST)."""
    from functools import wraps
    from django.http import JsonResponse
    from myapp.views import _dashboard_guard

    @wraps(view)
    def wrapper(request):
        if not _dashboard_guard(request):
            return JsonResponse({'status': 'error', 'detail': 'Staff only.'}, status=403)
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'detail': 'Invalid request method.'}, status=405)
        return view(request)
    return wrapper


_SNAPSHOT_FIELDS = ('id', 'name', 'publisher', 'description', 'tags', 'caps', 'kind', 'on_api',
                    'testable', 'calls_last_month', 'released', 'preview')
RESULT_ORDER = ('works', 'busy', 'error', 'no_access', 'retired', 'bad_key')


def _run_counts(run_ids):
    """{run_id: {'tested', 'works', 'no_access', 'retired', 'busy', 'error', 'fastest'}}"""
    from django.db.models import Count, Min, Q
    from myapp.models import APICheckResult
    rows = APICheckResult.objects.filter(run_id__in=run_ids).values('run_id').annotate(
        tested=Count('model_id', distinct=True),
        works=Count('id', filter=Q(status='works')),
        no_access=Count('id', filter=Q(status='no_access')),
        retired=Count('id', filter=Q(status='retired')),
        busy=Count('id', filter=Q(status='busy')),
        error=Count('id', filter=Q(status='error')),
        fastest_seconds=Min('seconds', filter=Q(status='works')),
    )
    counts = {row['run_id']: row for row in rows}
    fastest = {}
    for row in APICheckResult.objects.filter(run_id__in=run_ids, status='works').order_by('seconds').values('run_id', 'model_id', 'seconds'):
        fastest.setdefault(row['run_id'], row)
    for run_id, row in fastest.items():
        counts.setdefault(run_id, {})['fastest'] = row
    return counts


def dashboard_api_checker(request):
    from django.shortcuts import render
    from myapp.views import dashboard_staff_required

    @dashboard_staff_required
    def page(request):
        from myapp.models import APICheckRun
        if request.GET.get('refresh'):
            cache.delete(CACHE_KEY)
        models, info = build_models()
        runs = list(APICheckRun.objects.select_related('created_by').defer('models_snapshot')[:50])
        counts = _run_counts([run.pk for run in runs])
        for run in runs:
            run.counts = counts.get(run.pk, {})
        added = set()
        try:
            from myapp.models import CustomAIModel
            added = {m.lower() for m in CustomAIModel.objects.values_list('model_id', flat=True)}
        except Exception:
            pass
        for m in models:
            m['already_added'] = m['id'].lower() in added
        return render(request, 'dashboard/api_checker.html', {
            'active': 'api_checker',
            'models': models,
            'info': info,
            'summary': summary(models),
            'capability_labels': CAPABILITY_LABELS,
            'runs': runs,
            'can_see_keys': request.user.is_superuser,
        })
    return page(request)


def dashboard_api_checker_run(request, pk):
    from django.shortcuts import get_object_or_404, render
    from myapp.views import dashboard_staff_required

    @dashboard_staff_required
    def page(request):
        from myapp.models import APICheckRun
        run = get_object_or_404(APICheckRun.objects.select_related('created_by'), pk=pk)
        # The latest result per model (a model tested twice keeps its last answer).
        latest = {}
        for result in run.results.order_by('tested_at'):
            latest[result.model_id] = result
        results = sorted(latest.values(), key=lambda r: (
            RESULT_ORDER.index(r.status) if r.status in RESULT_ORDER else 99, r.seconds or 9e9, r.model_id))
        snapshot = run.models_snapshot or []
        by_id = {m.get('id'): m for m in snapshot}
        for result in results:
            result.model = by_id.get(result.model_id, {})
        counts = {status: sum(1 for r in results if r.status == status) for status in RESULT_ORDER}
        return render(request, 'dashboard/api_checker_run.html', {
            'active': 'api_checker',
            'run': run,
            'results': results,
            'counts': counts,
            'snapshot': snapshot,
            'snapshot_summary': run.summary or {},
            'capability_labels': CAPABILITY_LABELS,
            'can_see_keys': request.user.is_superuser,
        })
    return page(request)


def dashboard_api_checker_run_delete(request, pk):
    from django.contrib import messages
    from django.shortcuts import get_object_or_404, redirect
    from myapp.views import dashboard_staff_required

    @dashboard_staff_required
    def act(request):
        from myapp.models import APICheckRun
        run = get_object_or_404(APICheckRun, pk=pk)
        if request.method == 'POST':
            run.delete()
            messages.success(request, 'That check and its saved key were deleted from History.')
        return redirect('dashboard_api_checker')
    return act(request)


@_staff_json
def dashboard_api_checker_key(request):
    """Check a key and start a History entry for it, with a snapshot of
    every model as NVIDIA lists them right now."""
    from django.http import JsonResponse
    from myapp.models import APICheckRun
    models, info = build_models()
    raw_key = custom_models.clean_key(_json_body(request).get('key', ''))
    result = check_key(raw_key, models)
    if raw_key:
        run = APICheckRun.objects.create(
            created_by=request.user, api_key=raw_key[:300],
            key_status=result['status'], key_detail=(result.get('detail') or '')[:300],
            key_seconds=result.get('seconds'), key_model=result.get('model', ''),
            summary={**summary(models), 'catalog_ok': info.get('catalog_ok'), 'api_ok': info.get('api_ok')},
            models_snapshot=[{field: m.get(field) for field in _SNAPSHOT_FIELDS} for m in models],
        )
        result['run_id'] = run.pk
    return JsonResponse(result)


@_staff_json
def dashboard_api_checker_test(request):
    from django.http import JsonResponse
    data = _json_body(request)
    model_id = str(data.get('model') or '')
    models, _info = build_models()
    if not any(m['id'] == model_id and m['testable'] for m in models):
        return JsonResponse(_result('error', 'Only chat models on the NVIDIA API can be tested here.'))
    result = test_model(data.get('key', ''), model_id)
    run_id = data.get('run_id')
    if isinstance(run_id, int):
        from myapp.models import APICheckResult, APICheckRun
        if APICheckRun.objects.filter(pk=run_id).exists():
            APICheckResult.objects.create(
                run_id=run_id, model_id=model_id[:200], status=result['status'],
                seconds=result.get('seconds'), detail=(result.get('detail') or '')[:300],
                reply=(result.get('reply') or '')[:200],
            )
    return JsonResponse(result)
