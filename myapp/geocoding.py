"""Reverse geocoding (lat/lng -> "City, State") shared by the live
location-consent save (views.ai_location_update) and the background backfill
for profiles that granted location before that save started caching a name
(location_backfill.py)."""
import requests


def reverse_geocode_place(latitude, longitude):
    """Best-effort "City, State" for one lat/lng, via OpenStreetMap's free
    Nominatim reverse geocoder. Called once, when a user grants location (a
    rare event) or by the one-time backfill job, and the result is cached on
    the profile — the dashboard signups/user-data tables then just display
    that cached name instead of raw coordinates or a live per-page-load
    lookup (which used to silently stay stuck on coordinates whenever
    Nominatim rate-limited or blocked the server, since every dashboard page
    view fired its own request)."""
    try:
        response = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'format': 'jsonv2', 'lat': str(latitude), 'lon': str(longitude), 'zoom': 10},
            headers={'User-Agent': 'VidhyoraEduTrellis/1.0'},
            timeout=4,
        )
        response.raise_for_status()
        address = response.json().get('address') or {}
    except (requests.RequestException, ValueError):
        return ''
    locality = (
        address.get('city') or address.get('town') or address.get('village')
        or address.get('county') or address.get('state_district') or ''
    )
    state = address.get('state') or ''
    parts = [part for part in (locality, state) if part]
    return ', '.join(parts)
