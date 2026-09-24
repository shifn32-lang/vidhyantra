"""Resolve an outbound provider API key/URL, preferring a dashboard-saved
override (ProviderAPICredential) over the settings.py/env default. Every
live call site that used to read `getattr(settings, X, '')` for one of these
reads through get_key(X) instead, so a change saved on the dashboard's API
Keys page takes effect immediately (next request), no redeploy needed.

Cached briefly so normal chat/image traffic does not hit the database on
every request just to resolve a credential that almost never changes.
"""
from django.conf import settings
from django.core.cache import cache

_CACHE_PREFIX = 'provider_key:'
_CACHE_TTL = 20  # seconds
_UNSET = object()


def get_db_override(setting_name):
    """Just the dashboard-saved value for setting_name, or '' if nothing was
    ever saved (settings.py's own default is not consulted here) — lets a
    caller tell "no override exists" apart from "override happens to match
    the default", which get_key()'s combined return can't."""
    cache_key = _CACHE_PREFIX + setting_name
    override = cache.get(cache_key, _UNSET)
    if override is _UNSET:
        override = ''
        try:
            from myapp.models import ProviderAPICredential
            row = (
                ProviderAPICredential.objects
                .filter(setting_name=setting_name)
                .values_list('value', flat=True)
                .first()
            )
            if row:
                override = row.strip()
        except Exception:
            # Table not migrated yet, DB unavailable, etc. — fall through to
            # the settings.py default rather than breaking every AI request.
            override = ''
        cache.set(cache_key, override, _CACHE_TTL)
    return override


def get_key(setting_name, default=''):
    override = get_db_override(setting_name)
    return override or (getattr(settings, setting_name, default) or default)


def invalidate(setting_name):
    cache.delete(_CACHE_PREFIX + setting_name)
