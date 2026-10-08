"""Dashboard on/off switch and request counters for individual AI models (see
models.AIModelControl). Read on every chat request, so the enabled flag is
cached briefly; every write helper swallows its own errors because telemetry
must never be the reason a chat reply fails.
"""
import logging

from django.core.cache import cache
from django.db.models import F
from django.utils import timezone

logger = logging.getLogger(__name__)

_CACHE_PREFIX = 'model_enabled:'
_CACHE_TTL = 15  # seconds
_UNSET = object()


def is_enabled(model_key):
    cache_key = _CACHE_PREFIX + model_key
    enabled = cache.get(cache_key, _UNSET)
    if enabled is _UNSET:
        enabled = True
        try:
            from myapp.models import AIModelControl
            row = AIModelControl.objects.filter(model_key=model_key).values_list('is_enabled', flat=True).first()
            if row is not None:
                enabled = bool(row)
        except Exception:
            # Table not migrated yet or DB unavailable: stay enabled rather
            # than taking the model offline over a bookkeeping failure.
            enabled = True
        cache.set(cache_key, enabled, _CACHE_TTL)
    return enabled


def set_enabled(model_key, enabled):
    from myapp.models import AIModelControl
    AIModelControl.objects.update_or_create(model_key=model_key, defaults={'is_enabled': bool(enabled)})
    cache.delete(_CACHE_PREFIX + model_key)


def _bump(model_key, counter, **fields):
    """Increment one counter (atomically) and set any extra fields, creating
    the row on first use."""
    try:
        from myapp.models import AIModelControl
        AIModelControl.objects.get_or_create(model_key=model_key)
        AIModelControl.objects.filter(model_key=model_key).update(**{counter: F(counter) + 1}, **fields)
    except Exception:
        logger.debug('Could not record %s for %s', counter, model_key, exc_info=True)


def record_request(model_key):
    _bump(model_key, 'request_count', last_used_at=timezone.now())


def record_success(model_key):
    _bump(model_key, 'success_count')


def record_error(model_key, message):
    _bump(model_key, 'error_count', last_error=(message or '')[:300])


def record_test(model_key, ok, message):
    from myapp.models import AIModelControl
    AIModelControl.objects.update_or_create(
        model_key=model_key,
        defaults={'last_tested_at': timezone.now(), 'last_test_ok': ok, 'last_test_message': (message or '')[:300]},
    )


_TEXT_PREFIX = 'model_text:'


def get_text(model_key):
    """(display_name, description) saved on the dashboard; '' for either one
    means the built-in default applies."""
    cache_key = _TEXT_PREFIX + model_key
    value = cache.get(cache_key, _UNSET)
    if value is _UNSET:
        value = ('', '')
        try:
            from myapp.models import AIModelControl
            row = AIModelControl.objects.filter(model_key=model_key).values_list('display_name', 'description').first()
            if row:
                value = (row[0] or '', row[1] or '')
        except Exception:
            value = ('', '')
        cache.set(cache_key, value, _CACHE_TTL)
    return value


def set_text(model_key, display_name, description):
    from myapp.models import AIModelControl
    AIModelControl.objects.update_or_create(
        model_key=model_key,
        defaults={'display_name': display_name.strip(), 'description': description.strip()},
    )
    cache.delete(_TEXT_PREFIX + model_key)


_REMOVED_PREFIX = 'model_removed:'


def is_removed(model_key):
    """Removed on API Settings: hidden from the picker until restored."""
    cache_key = _REMOVED_PREFIX + model_key
    removed = cache.get(cache_key, _UNSET)
    if removed is _UNSET:
        try:
            from myapp.models import AIModelControl
            removed = bool(AIModelControl.objects.filter(model_key=model_key, is_removed=True).exists())
        except Exception:
            removed = False
        cache.set(cache_key, removed, _CACHE_TTL)
    return removed


def set_removed(model_key, removed):
    """Remove (switch off and hide) or restore (switch back on) a model."""
    from myapp.models import AIModelControl
    AIModelControl.objects.update_or_create(
        model_key=model_key, defaults={'is_removed': bool(removed), 'is_enabled': not removed},
    )
    cache.delete(_REMOVED_PREFIX + model_key)
    cache.delete(_CACHE_PREFIX + model_key)


IDENTITY_FIELDS = ('name', 'creator', 'model', 'notes')
_IDENTITY_PREFIX = 'model_identity:'


def get_identity(model_key):
    """{'name', 'creator', 'model', 'notes'} with only the fields the site
    owner filled in, or {} when the built-in identity rules apply."""
    cache_key = _IDENTITY_PREFIX + str(model_key)
    value = cache.get(cache_key, _UNSET)
    if value is _UNSET:
        value = {}
        try:
            from myapp.models import AIModelControl
            row = AIModelControl.objects.filter(model_key=model_key).values_list(
                'identity_name', 'identity_creator', 'identity_model', 'identity_notes').first()
            if row:
                value = {field: text.strip() for field, text in zip(IDENTITY_FIELDS, row) if (text or '').strip()}
        except Exception:
            value = {}
        cache.set(cache_key, value, _CACHE_TTL)
    return value


def set_identity(model_key, name='', creator='', model='', notes=''):
    from myapp.models import AIModelControl
    AIModelControl.objects.update_or_create(model_key=model_key, defaults={
        'identity_name': name.strip()[:80], 'identity_creator': creator.strip()[:120],
        'identity_model': model.strip()[:120], 'identity_notes': notes.strip()[:1000],
    })
    cache.delete(_IDENTITY_PREFIX + model_key)

