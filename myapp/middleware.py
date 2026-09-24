from django.http import HttpResponsePermanentRedirect
from django.shortcuts import render

from django.contrib.auth import logout
from django.contrib.sessions.models import Session
from django.utils import timezone

from .models import ActiveUserSession
from .single_device import register_active_session


class SiteDisabledMiddleware:
    """The dashboard Customize page's "Disable site" checkbox (see
    SiteCustomization.site_disabled) — every visitor gets a bare 404 instead
    of the real site. The dashboard and Django admin stay reachable
    (by path, not by login state) so staff can always get back in and
    switch it off again, even logged out, without touching the database
    directly. Static/media stay reachable too so the dashboard's own login
    page still renders correctly while the rest of the site is down."""

    EXEMPT_PREFIXES = ('/store/dashboard/', '/admin/', '/static/', '/media/')

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith(self.EXEMPT_PREFIXES):
            return self.get_response(request)
        from myapp.models import SiteCustomization
        try:
            disabled = SiteCustomization.get_solo().site_disabled
        except Exception:
            disabled = False
        if disabled:
            return render(request, 'site_disabled.html', status=404)
        return self.get_response(request)


class CanonicalHostMiddleware:
    """Consolidate the bare domain onto the canonical www hostname."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.get_host().partition(':')[0].lower() == 'edutrellis.in':
            return HttpResponsePermanentRedirect(
                f'https://www.edutrellis.in{request.get_full_path()}'
            )
        return self.get_response(request)


class HideAdminFromNonStaffMiddleware:
    """Django's built-in admin login view tells an authenticated non-staff
    user "You are authenticated as X, but are not authorized to access this
    page" — which both confirms /admin/ exists and echoes their email back
    to them. Anyone logged in but not staff/superuser (e.g. an
    AI-premium-only account) should see the same 404 as any unknown URL
    instead, same as an anonymous visitor poking at random paths."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/admin/') and request.user.is_authenticated and not (request.user.is_staff or request.user.is_superuser):
            from myapp.views import custom_404
            return custom_404(request)
        return self.get_response(request)


class SingleDeviceSessionMiddleware:
    """Log out a browser when a newer login has claimed the same account."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            current_key = request.session.session_key
            active = ActiveUserSession.objects.filter(user=request.user).first()

            if not current_key:
                request.session.save()
                register_active_session(request.user, request.session.session_key)
            elif active is None:
                # Covers accounts that were already logged in when this
                # feature was deployed: their current browser gets the slot.
                register_active_session(request.user, current_key)
            elif active.session_key != current_key:
                active_session_still_valid = Session.objects.filter(
                    session_key=active.session_key,
                    expire_date__gt=timezone.now(),
                ).exists()
                if active_session_still_valid:
                    # The active key belongs to the newer device. Django's
                    # logout also clears this browser's stale auth cookie.
                    logout(request)
                else:
                    # A normal session-key rotation (for example after a
                    # password change) or an expired record may leave the
                    # database pointer stale; safely let the only live
                    # authenticated session reclaim it.
                    register_active_session(request.user, current_key)

        return self.get_response(request)


class PublicAssetCacheMiddleware:
    """Give version-stable static/media files a useful browser cache.

    This middleware sits before WhiteNoise so it also wraps files served
    directly by WhiteNoise rather than only normal Django view responses.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if response.status_code == 200:
            if request.path.startswith('/static/'):
                response['Cache-Control'] = 'public, max-age=86400, stale-while-revalidate=604800'
            elif request.path.startswith((
                '/media/products/', '/media/categories/', '/media/about/', '/media/pwa/',
            )):
                response['Cache-Control'] = 'public, max-age=604800, stale-while-revalidate=2592000'
        return response
