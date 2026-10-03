"""Dashboard → OpenCode Data: who uses the Start coding (OpenCode) backend,
how much, and what was said, in the same style as AI Activity.

    /store/dashboard/opencode/                 users, with their usage
    /store/dashboard/opencode/user/<id>/       one user's computers and sessions
    /store/dashboard/opencode/session/<id>/    the requests and replies in a session
"""
from datetime import datetime, timedelta, timezone as dt_timezone

from django.contrib import messages
from django.contrib.auth.models import User
from django.db.models import Count, Max, Q, Sum
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from myapp import coding_api, model_controls
from myapp.models import AICodingKey, AICodingRequest, AICodingSession
from myapp.views import AI_ACTIVITY_DATE_FILTERS, dashboard_staff_required

USER_LIMIT = 300
SESSION_REQUEST_LIMIT = 500


def _since(date_filter):
    """(start, end) for a "last active" filter, or (None, None) for all time."""
    now = timezone.localtime()
    if date_filter == 'yesterday':
        start = AI_ACTIVITY_DATE_FILTERS['yesterday'](now)
        return start, start + timedelta(days=1)
    if date_filter in AI_ACTIVITY_DATE_FILTERS:
        return AI_ACTIVITY_DATE_FILTERS[date_filter](now), None
    return None, None


def _display_name(user):
    return (f'{user.first_name} {user.last_name}'.strip()) or user.username


@dashboard_staff_required
def dashboard_opencode(request):
    q = request.GET.get('q', '').strip()
    date_filter = request.GET.get('when', '').strip()
    if date_filter not in ('today', 'yesterday', 'week', 'month'):
        date_filter = 'all'
    sort = request.GET.get('sort', '').strip()
    if sort not in ('recent', 'most_requests', 'most_sessions'):
        sort = 'recent'

    today_start = AI_ACTIVITY_DATE_FILTERS['today'](timezone.localtime())
    usage = {
        row['user_id']: row for row in AICodingRequest.objects.values('user_id').annotate(
            requests=Count('id'), sessions=Count('session', distinct=True), last=Max('created_at'),
            errors=Count('id', filter=Q(status=AICodingRequest.STATUS_ERROR)),
            prompt_tokens=Sum('prompt_tokens'), completion_tokens=Sum('completion_tokens'),
        )
    }
    today = dict(
        AICodingRequest.objects.filter(created_at__gte=today_start)
        .values('user_id').annotate(n=Count('id')).values_list('user_id', 'n')
    )
    computers = {}
    for key in AICodingKey.objects.order_by('created_at'):
        computers.setdefault(key.user_id, []).append(key.label or 'manual key')

    users = User.objects.filter(pk__in=set(usage) | set(computers))
    if q:
        users = users.filter(
            Q(email__icontains=q) | Q(username__icontains=q) | Q(first_name__icontains=q) | Q(last_name__icontains=q)
        )
    start, end = _since(date_filter)
    rows = []
    for user in users:
        row = usage.get(user.pk, {})
        last = row.get('last')
        if start and (not last or last < start or (end and last >= end)):
            continue
        tokens = (row.get('prompt_tokens') or 0) + (row.get('completion_tokens') or 0)
        rows.append({
            'user': user, 'name': _display_name(user),
            'computers': computers.get(user.pk, []),
            'requests': row.get('requests', 0), 'today': today.get(user.pk, 0),
            'sessions': row.get('sessions', 0), 'errors': row.get('errors', 0),
            'tokens': tokens, 'last': last,
        })
    floor = datetime(2000, 1, 1, tzinfo=dt_timezone.utc)
    sorters = {
        'recent': lambda r: r['last'] or floor,
        'most_requests': lambda r: r['requests'],
        'most_sessions': lambda r: r['sessions'],
    }
    rows.sort(key=sorters[sort], reverse=True)

    day_ago = timezone.now() - timedelta(days=1)
    context = {
        'active': 'opencode',
        'rows': rows[:USER_LIMIT], 'row_count': len(rows),
        'q': q, 'date_filter': date_filter, 'sort': sort,
        'enabled': model_controls.is_enabled(coding_api.CONTROL_KEY),
        'stats': {
            'users': len(set(computers) | set(usage)),
            'requests_total': AICodingRequest.objects.count(),
            'requests_today': AICodingRequest.objects.filter(created_at__gte=today_start).count(),
            'sessions': AICodingSession.objects.count(),
            'errors_day': AICodingRequest.objects.filter(created_at__gte=day_ago, status=AICodingRequest.STATUS_ERROR).count(),
        },
    }
    return render(request, 'dashboard/opencode.html', context)


@dashboard_staff_required
def dashboard_opencode_user(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    sessions = list(
        AICodingSession.objects.filter(user=user).annotate(
            errors=Count('requests', filter=Q(requests__status=AICodingRequest.STATUS_ERROR)),
            prompt=Sum('requests__prompt_tokens'), completion=Sum('requests__completion_tokens'),
        ).order_by('-last_request_at')[:200]
    )
    for item in sessions:
        item.tokens = (item.prompt or 0) + (item.completion or 0)
    totals = AICodingRequest.objects.filter(user=user).aggregate(
        requests=Count('id'), last=Max('created_at'),
        errors=Count('id', filter=Q(status=AICodingRequest.STATUS_ERROR)),
        prompt=Sum('prompt_tokens'), completion=Sum('completion_tokens'),
    )
    context = {
        'active': 'opencode',
        'target': user, 'name': _display_name(user),
        'devices': AICodingKey.objects.filter(user=user).order_by('-created_at'),
        'sessions': sessions, 'totals': totals,
        'tokens': (totals['prompt'] or 0) + (totals['completion'] or 0),
        'has_access': coding_api.has_access(user),
    }
    return render(request, 'dashboard/opencode_user.html', context)


@dashboard_staff_required
def dashboard_opencode_session(request, pk):
    session = get_object_or_404(AICodingSession.objects.select_related('user'), pk=pk)
    requests_qs = session.requests.order_by('created_at')
    total = requests_qs.count()
    context = {
        'active': 'opencode',
        'session': session, 'target': session.user, 'name': _display_name(session.user),
        'requests': list(requests_qs[:SESSION_REQUEST_LIMIT]), 'request_total': total,
        'clipped': total > SESSION_REQUEST_LIMIT,
    }
    return render(request, 'dashboard/opencode_session.html', context)


@dashboard_staff_required
def dashboard_opencode_session_delete(request, pk):
    session = get_object_or_404(AICodingSession, pk=pk)
    if request.method == 'POST':
        user_id = session.user_id
        session.delete()
        messages.success(request, 'That session and its requests were deleted.')
        return redirect('dashboard_opencode_user', user_id=user_id)
    return redirect('dashboard_opencode_session', pk=pk)
