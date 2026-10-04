"""The panels on the dashboard's API Settings page: one per ChatGPT model,
image generation and web search. Each panel can hold its API
key, run a live connection test, switch the feature on/off, show how many
requests it has served, and (for models) rename it / change its description.

State lives in two places that the rest of the app already reads:
ProviderAPICredential (the key, via myapp.provider_keys) and AIModelControl
(on/off, counters, name/description, last test, via myapp.model_controls).
"""
from myapp import ai_chat, coding_api, model_controls
from myapp.provider_keys import get_key, invalidate

KIND_TEXT = 'text'
KIND_IMAGE = 'image'
KIND_SEARCH = 'search'

_ENABLE_TEXT = {
    KIND_TEXT: 'show {label} in the model picker and let it answer requests',
    KIND_IMAGE: 'allow image generation (image requests from every model)',
    KIND_SEARCH: 'let the AI use live web search for current-information questions',
}


def definitions():
    from myapp.web_search import SEARCH_CONTROL_KEY
    nvidia = 'NVIDIA API key (nvapi-…)'
    return [
        {'id': 'chat', 'kind': KIND_TEXT, 'model_key': ai_chat.CHAT_CONTROL_KEY, 'test_model': 'quick',
         'title': 'Vidhyora Chat (Ultra / Quick / Code)', 'toggle': False,
         'text_models': list(ai_chat.CHAT_GROUP_KEYS),
         'note': 'One NVIDIA key runs Vidhyora Ultra, Quick and Code, and the workers the ChatGPT '
                 'models route through. It cannot be switched off.',
         'keys': [('NVIDIA_API_KEY', nvidia)]},
        {'id': 'luna', 'kind': KIND_TEXT, 'model_key': ai_chat.CHATGPT_56_MODEL_KEY,
         'note': 'The automatic route: picks the best worker for each message.',
         'keys': [('NVIDIA_LUNA_API_KEY', nvidia)]},
        {'id': 'sol', 'kind': KIND_TEXT, 'model_key': ai_chat.SOL_MODEL_KEY,
         'note': 'Also the default model for staff and subscribers.',
         'keys': [('NVIDIA_NEMOTRON_SUPER_API_KEY', nvidia)]},
        {'id': 'terra', 'kind': KIND_TEXT, 'model_key': ai_chat.TERRA_MODEL_KEY,
         'note': '', 'keys': [('NVIDIA_TERRA_API_KEY', nvidia)]},
        {'id': 'coding', 'kind': KIND_TEXT, 'model_key': coding_api.CONTROL_KEY, 'test_model': 'code',
         'title': 'Start Coding (OpenCode CLI)', 'keys': [],
         'enable_text': 'let users connect the OpenCode terminal agent to Vidhyora Code (Start coding in the account menu)',
         'note': 'Powers the terminal coding agent through /api/v1/code/ using the Vidhyora Chat key above — '
                 'no separate key. Users sign in with a personal key made under Start coding; paid plans and staff only. '
                 'Test connection checks the Vidhyora Code model that serves it.'},
        {'id': 'image', 'kind': KIND_IMAGE, 'model_key': ai_chat.FLUX_KLEIN_4B_MODEL_KEY,
         'note': 'Text-to-image and photo editing.', 'keys': [('NVIDIA_FLUX_API_KEY', nvidia)]},
        {'id': 'search', 'kind': KIND_SEARCH, 'model_key': SEARCH_CONTROL_KEY,
         'title': 'Web Search (Tavily)', 'note': 'Live results the AI can pull into its replies.',
         'keys': [('TAVILY_API_KEY', 'Tavily API key (tvly-…)')]},
    ]


def _mask(value):
    value = (value or '').strip()
    if not value:
        return ''
    if len(value) <= 8:
        return '•' * len(value)
    return value[:4] + '…' + value[-4:]


def _find(panel_id):
    return next((d for d in definitions() if d['id'] == panel_id), None)


def run_test(defn):
    """(ok, message) from a live request against the panel's provider."""
    kind = defn['kind']
    if kind == KIND_TEXT:
        return ai_chat.test_model_connection(defn.get('test_model') or defn['model_key'])
    if kind == KIND_IMAGE:
        from myapp import image_generation
        return image_generation.test_connection()
    from myapp import web_search
    return web_search.test_connection()


def panel_context():
    """Everything the template needs to draw every panel."""
    from myapp.models import AIModelControl, ProviderAPICredential
    ai_chat.apply_model_text_overrides()
    controls = {row.model_key: row for row in AIModelControl.objects.all()}
    overrides = {row.setting_name: row for row in ProviderAPICredential.objects.all()}
    panels = []
    for defn in definitions():
        key = defn['model_key']
        cfg = ai_chat.MODELS.get(key)
        text_keys = defn.get('text_models') or ([key] if key in ai_chat._MODEL_TEXT_DEFAULTS else [])
        items = []
        for text_key in text_keys:
            default_name, default_description = ai_chat.model_default_text(text_key)
            items.append({
                'key': text_key,
                'label': ai_chat.MODELS[text_key]['label'],
                'description': ai_chat.MODELS[text_key]['description'],
                'default_name': default_name,
                'default_description': default_description,
            })
        title = defn.get('title') or (cfg['label'] if cfg else '')
        fields = []
        for setting, label in defn['keys']:
            current = get_key(setting)
            override = overrides.get(setting)
            fields.append({
                'setting': setting, 'label': label,
                'configured': bool(current), 'preview': _mask(current),
                'overridden': override is not None,
                'updated_at': override.updated_at if override else None,
            })
        panels.append({
            'id': defn['id'], 'kind': defn['kind'], 'title': title, 'note': defn['note'],
            'fields': fields, 'configured': all(f['configured'] for f in fields),
            'toggle': defn.get('toggle', True),
            'enabled': model_controls.is_enabled(key),
            'enable_text': defn.get('enable_text') or _ENABLE_TEXT[defn['kind']].format(label=title),
            'control': controls.get(key),
            'items': items,
        })
    return panels


def handle_post(request):
    """Apply one panel's form. Returns (panel_id, ok, message)."""
    from myapp.models import ProviderAPICredential
    defn = _find(request.POST.get('panel', ''))
    action = request.POST.get('action', '')
    if defn is None:
        return '', True, ''
    panel_id, key = defn['id'], defn['model_key']
    settings_for_panel = {setting for setting, _label in defn['keys']}

    if action.startswith('clear:'):
        setting = action.split(':', 1)[1]
        if setting in settings_for_panel:
            ProviderAPICredential.objects.filter(setting_name=setting).delete()
            invalidate(setting)
            return panel_id, True, 'Saved key removed; reverted to the server default.'
        return panel_id, True, ''

    parts = []
    new_key_saved = False
    if action == 'save':
        for setting, _label in defn['keys']:
            value = request.POST.get(f'key__{setting}', '').strip()
            if value:
                ProviderAPICredential.objects.update_or_create(
                    setting_name=setting,
                    defaults={'value': value, 'updated_by': request.user},
                )
                invalidate(setting)
                new_key_saved = True
        if new_key_saved:
            parts.append('API key saved.')

        text_changed = False
        for text_key in defn.get('text_models') or ([key] if key in ai_chat._MODEL_TEXT_DEFAULTS else []):
            if f'name__{text_key}' not in request.POST:
                continue
            default_name, default_description = ai_chat.model_default_text(text_key)
            name = request.POST.get(f'name__{text_key}', '').strip()[:60]
            description = request.POST.get(f'description__{text_key}', '').strip()[:300]
            # Text equal to the built-in default is stored as "no override".
            name = '' if name == default_name else name
            description = '' if description == default_description else description
            if (name, description) != model_controls.get_text(text_key):
                model_controls.set_text(text_key, name, description)
                text_changed = True
        if text_changed:
            ai_chat.apply_model_text_overrides()
            parts.append('Name and description saved.')

        if defn.get('toggle', True):
            enabled = request.POST.get('enabled') == 'on'
            if enabled != model_controls.is_enabled(key):
                parts.append('Enabled.' if enabled else 'Disabled.')
            model_controls.set_enabled(key, enabled)
    elif action != 'test':
        return panel_id, True, ''

    # Test after a new key is pasted (so "saved" always comes with "does it
    # work") and whenever the Test button is pressed.
    if action == 'test' or new_key_saved:
        ok, detail = run_test(defn)
        model_controls.record_test(key, ok, detail)
        parts.append(detail if ok else f'Test failed: {detail}')
        return panel_id, ok, ' '.join(parts)
    return panel_id, True, ' '.join(parts) or 'No changes.'
