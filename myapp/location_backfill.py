"""One-time background backfill: resolve a "City, State" name for every
StoreProfile that granted location before ai_location_update started caching
one (see myapp.geocoding), so old rows stop showing raw coordinates in the
Signups / User Data dashboard tables. Runs as a background thread, not
inline in a request — OpenStreetMap's free Nominatim geocoder asks for at
most 1 request/second, so resolving a few hundred rows takes minutes, far
longer than any request should block for."""
import threading
import time

from django.core.cache import cache

from myapp.geocoding import reverse_geocode_place

_STATUS_CACHE_KEY = 'location_backfill_status'
_STATUS_TTL = 3600
_REQUEST_INTERVAL_SECONDS = 1.1


def _set_status(**fields):
    status = cache.get(_STATUS_CACHE_KEY) or {}
    status.update(fields)
    cache.set(_STATUS_CACHE_KEY, status, _STATUS_TTL)


def get_status():
    return cache.get(_STATUS_CACHE_KEY) or {'running': False, 'total': 0, 'done': 0, 'resolved': 0}


def _run():
    from django.db import close_old_connections
    from myapp.models import StoreProfile
    close_old_connections()
    try:
        pending = StoreProfile.objects.filter(
            location_consent=StoreProfile.LOCATION_GRANTED,
            location_place_name='',
            location_latitude__isnull=False,
            location_longitude__isnull=False,
        ).only('pk', 'location_latitude', 'location_longitude')
        total = pending.count()
        _set_status(running=True, total=total, done=0, resolved=0)
        done = 0
        resolved = 0
        for profile in pending.iterator():
            name = reverse_geocode_place(profile.location_latitude, profile.location_longitude)
            if name:
                StoreProfile.objects.filter(pk=profile.pk).update(location_place_name=name)
                resolved += 1
            done += 1
            _set_status(running=True, total=total, done=done, resolved=resolved)
            time.sleep(_REQUEST_INTERVAL_SECONDS)
    finally:
        status = cache.get(_STATUS_CACHE_KEY) or {}
        status['running'] = False
        cache.set(_STATUS_CACHE_KEY, status, _STATUS_TTL)
        close_old_connections()


def start():
    """Returns False without doing anything if a run is already in progress."""
    status = cache.get(_STATUS_CACHE_KEY)
    if status and status.get('running'):
        return False
    cache.set(_STATUS_CACHE_KEY, {'running': True, 'total': 0, 'done': 0, 'resolved': 0}, _STATUS_TTL)
    threading.Thread(target=_run, daemon=True, name='location-name-backfill').start()
    return True
