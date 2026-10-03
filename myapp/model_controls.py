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
