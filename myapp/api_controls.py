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
    ] + _custom_definitions()


def _custom_definitions():
    """One panel per model added with "Add a new model"."""
    from myapp.models import CustomAIModel
    try:
        rows = list(CustomAIModel.objects.all())
    except Exception:
        return []
    panels = []
    for row in rows:
        nvidia = row.api_type == CustomAIModel.API_NVIDIA
        panels.append({
            'id': row.model_key, 'kind': KIND_TEXT, 'model_key': row.model_key,
            'title': row.display_name, 'custom': row,
            'note': f"{'NVIDIA NIM' if nvidia else row.base_url} · {row.model_id}",
            'keys': [(row.key_setting, 'NVIDIA API key (nvapi-…)' if nvidia else 'API key')],
        })
    return panels


_PERSONA_KEYS = (ai_chat.CHATGPT_56_MODEL_KEY, ai_chat.SOL_MODEL_KEY, ai_chat.TERRA_MODEL_KEY)


def _chat_model_keys(defn):
    """The picker models a panel controls, whose identity can be set."""
    if defn.get('custom'):
        return [defn['model_key']]
    if defn['kind'] != KIND_TEXT or defn['id'] == 'coding':
        return []
    return list(defn.get('text_models') or [defn['model_key']])


def _identity_defaults(key):
    """What the model says about itself when nothing is filled in."""
    label = ai_chat.MODELS.get(key, {}).get('label', key)
    if key in _PERSONA_KEYS:
        return {'name': label, 'creator': 'OpenAI', 'model': 'GPT-5.6'}
    return {'name': label, 'creator': f'the {ai_chat.get_ai_brand_name()} team', 'model': label}


def _removable_keys(defn):
    if defn.get('custom'):
        return []
    return [key for key in (defn.get('text_models') or [defn['model_key']]) if key in ai_chat.REMOVABLE_MODEL_KEYS]


def _mask(value):
    value = (value or '').strip()
    if not value:
        return ''
    if len(value) <= 8:
        return '•' * len(value)
    return value[:4] + '…' + value[-4:]


def _find(panel_id):
    return next((d for d in definitions() if d['id'] == panel_id), None)


def _check_custom(row, api_key):
    """Run every check for an added model; keep what they found on the row.
    Returns (ok, report)."""
    from myapp import custom_models
    ok, report, found = custom_models.run_checks(row.api_type, row.base_url, row.model_id, api_key)
    if ok:
        row.thinking_control = found['thinking_control']
        row.vision = found['vision']
        row.model_id = found.get('model_id', row.model_id)
    custom_models.save_report(row, report)
    row.save()
    custom_models.mark_changed()
    return ok, report


def _report_summary(ok, report):
    failed = next((item for item in report if item['status'] == 'fail'), None)
    if failed:
        return f"{failed['label']}: {failed['detail']}"
    warned = [item['detail'] for item in report if item['status'] == 'warn']
    return 'All checks passed.' + (' ' + ' '.join(warned) if warned else '')


def run_test(defn):
    """(ok, message) from a live request against the panel's provider."""
    if defn.get('custom'):
        row = defn['custom']
        ok, report = _check_custom(row, get_key(row.key_setting))
        return ok, _report_summary(ok, report)
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
            'custom': defn.get('custom'),
            'identities': [
                {'key': chat_key, 'label': ai_chat.MODELS.get(chat_key, {}).get('label', chat_key),
                 'values': model_controls.get_identity(chat_key), 'defaults': _identity_defaults(chat_key)}
                for chat_key in _chat_model_keys(defn)
            ],
            'removable': [
                {'key': rkey, 'label': ai_chat.MODELS[rkey]['label'], 'removed': model_controls.is_removed(rkey)}
                for rkey in _removable_keys(defn)
            ],
        })
    return panels


def split_removed(panels):
    """(panels to show, removed models). A panel whose own model was removed
    leaves the list; a removed Vidhyora Ultra/Code stays in its shared panel
    but is listed under Removed models too."""
    shown, removed = [], []
    for panel in panels:
        own = next((r for r in panel['removable'] if r['key'] == panel_model_key(panel)), None)
        if own and own['removed']:
            removed.append({'panel': panel['id'], 'key': own['key'], 'label': own['label']})
            continue
        removed.extend({'panel': panel['id'], 'key': r['key'], 'label': r['label']} for r in panel['removable'] if r['removed'])
        shown.append(panel)
    return shown, removed


def panel_model_key(panel):
    defn = _find(panel['id'])
    return defn['model_key'] if defn else None


def handle_post(request):
    """Apply one panel's form. Returns (panel_id, ok, message)."""
    from myapp.models import ProviderAPICredential
    defn = _find(request.POST.get('panel', ''))
    action = request.POST.get('action', '')
    if defn is None:
        return '', True, ''
    panel_id, key = defn['id'], defn['model_key']
    settings_for_panel = {setting for setting, _label in defn['keys']}
    if defn.get('custom'):
        return _handle_custom_post(request, defn, action)

    if action.startswith(('remove:', 'restore:')):
        verb, model_key = action.split(':', 1)
        if model_key not in _removable_keys(defn):
            return panel_id, True, ''
        label = ai_chat.MODELS[model_key]['label']
        model_controls.set_removed(model_key, verb == 'remove')
        if verb == 'remove':
            return ('' if model_key == key else panel_id), True, (
                f'{label} was removed from the model picker. It is listed under Removed models, where you can restore it.')
        return panel_id, True, f'{label} is back in the model picker and switched on.'

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
        if _save_identities(request, defn):
            parts.append('Identity saved.')

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


def _handle_custom_post(request, defn, action):
    """Save / test / delete for a model added with "Add a new model". A new
    key is only kept when the checks pass with it."""
    from myapp import custom_models
    from myapp.models import AIModelControl, ProviderAPICredential
    row, panel_id = defn['custom'], defn['id']

    if action == 'delete':
        ProviderAPICredential.objects.filter(setting_name=row.key_setting).delete()
        invalidate(row.key_setting)
        AIModelControl.objects.filter(model_key=row.model_key).delete()
        name = row.display_name
        row.delete()
        custom_models.mark_changed()
        return '', True, f'{name} was removed from the model picker.'
    if action == 'test':
        ok, message = run_test(defn)
        model_controls.record_test(row.model_key, ok, message)
        return panel_id, ok, message
    if action != 'save':
        return panel_id, True, ''

    parts = []
    if _save_identities(request, defn):
        parts.append('Identity saved.')
    name = request.POST.get('custom_name', '').strip()[:60] or custom_models.default_name(row.model_id)
    description = request.POST.get('custom_description', '').strip()[:300]
    if (name, description) != (row.display_name, row.description):
        row.display_name, row.description = name, description
        row.save(update_fields=['display_name', 'description'])
        custom_models.mark_changed()
        parts.append('Name and description saved.')

    new_key = custom_models.clean_key(request.POST.get(f'key__{row.key_setting}', ''))
    if new_key:
        ok, report = _check_custom(row, new_key)
        if not ok:
            return panel_id, False, 'The new key was not saved. ' + _report_summary(ok, report)
        ProviderAPICredential.objects.update_or_create(
            setting_name=row.key_setting, defaults={'value': new_key, 'updated_by': request.user},
        )
        invalidate(row.key_setting)
        model_controls.record_test(row.model_key, True, _report_summary(ok, report))
        parts.append('New API key saved — all checks passed.')

    enabled = request.POST.get('enabled') == 'on'
    if enabled != model_controls.is_enabled(row.model_key):
        parts.append('Enabled.' if enabled else 'Disabled.')
    model_controls.set_enabled(row.model_key, enabled)
    return panel_id, True, ' '.join(parts) or 'No changes.'


def add_model(request):
    """The "Add a new model" form. Returns a dict for the template: ok,
    message, report (the checks) and values (to refill the form on failure)."""
    from myapp import custom_models
    from myapp.models import CustomAIModel, ProviderAPICredential
    api_type = request.POST.get('api_type', CustomAIModel.API_NVIDIA)
    if api_type not in dict(CustomAIModel.API_TYPE_CHOICES):
        api_type = CustomAIModel.API_NVIDIA
    nvidia = api_type == CustomAIModel.API_NVIDIA
    values = {
        'api_type': api_type,
        'base_url': '' if nvidia else custom_models.clean_base_url(request.POST.get('base_url', '')),
        'model_id': custom_models.clean_model_id(request.POST.get('model_id', '')),
        'display_name': request.POST.get('display_name', '').strip()[:60],
        'description': request.POST.get('description', '').strip()[:300],
    }
    api_key = custom_models.clean_key(request.POST.get('api_key', ''))

    duplicate = CustomAIModel.objects.filter(
        api_type=api_type, base_url=values['base_url'], model_id__iexact=values['model_id'],
    ).first() if values['model_id'] else None
    if duplicate:
        return {'ok': False, 'values': values, 'report': [],
                'message': f'{values["model_id"]} is already added as "{duplicate.display_name}". '
                           'Change its key or test it in its own section below.'}

    ok, report, found = custom_models.run_checks(api_type, values['base_url'], values['model_id'], api_key)
    if not ok:
        return {'ok': False, 'values': values, 'report': report,
                'message': 'The model was not added — fix the item marked below and try again.'}

    model_id = found.get('model_id', values['model_id'])
    row = CustomAIModel(
        api_type=api_type, base_url=values['base_url'], model_id=model_id,
        display_name=values['display_name'] or custom_models.default_name(model_id),
        description=values['description'],
        thinking_control=found['thinking_control'], vision=found['vision'],
        created_by=request.user,
    )
    custom_models.save_report(row, report)
    row.save()
    ProviderAPICredential.objects.update_or_create(
        setting_name=row.key_setting, defaults={'value': api_key, 'updated_by': request.user},
    )
    invalidate(row.key_setting)
    model_controls.set_enabled(row.model_key, True)
    model_controls.record_test(row.model_key, True, _report_summary(True, report))
    custom_models.mark_changed()
    return {'ok': True, 'values': {}, 'report': report, 'new_panel': row.model_key,
            'message': f'{row.display_name} was added and is now in the model picker for staff and subscribers.'}


def _save_identities(request, defn):
    """Save the "How it describes itself" fields of a panel's models.
    Returns True when anything changed."""
    changed = False
    for chat_key in _chat_model_keys(defn):
        if f'identity_name__{chat_key}' not in request.POST:
            continue
        values = {field: request.POST.get(f'identity_{field}__{chat_key}', '').strip()
                  for field in model_controls.IDENTITY_FIELDS}
        current = model_controls.get_identity(chat_key)
        if {k: v for k, v in values.items() if v} != current:
            model_controls.set_identity(chat_key, **values)
            changed = True
    return changed

