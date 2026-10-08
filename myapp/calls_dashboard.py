"""Dashboard → Call Data: every voice call made with the phone button on the
/AI/ page — who called, when, for how long, the recording and the full
transcript — with search and filters, in the same style as AI Activity.

    /store/dashboard/calls/                      all calls, filterable
    /store/dashboard/calls/<id>/                 one call: player and transcript
    /store/dashboard/calls/<id>/audio/           its recording (seekable)
    /store/dashboard/calls/<id>/download/<fmt>/  transcript as txt or pdf
    /store/dashboard/calls/<id>/delete/          POST
"""
from datetime import timedelta

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q, Sum
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from myapp import ai_calls
from myapp.models import AICall
from myapp.views import AI_ACTIVITY_DATE_FILTERS, dashboard_staff_required

PAGE_SIZE = 50

LENGTH_FILTERS = {
    'short': Q(duration_seconds__lt=60),
    'medium': Q(duration_seconds__gte=60, duration_seconds__lt=300),
    'long': Q(duration_seconds__gte=300),
}
SORTS = {
    'recent': '-started_at',
    'oldest': 'started_at',
    'longest': '-duration_seconds',
    'most_lines': '-turn_count',
}


def _display_name(user):
    return (f'{user.first_name} {user.last_name}'.strip()) or user.username


def duration_text(seconds):
    minutes, secs = divmod(int(seconds or 0), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes:02d}m'
    if minutes:
        return f'{minutes}m {secs:02d}s'
    return f'{secs}s'


@dashboard_staff_required
def dashboard_calls(request):
    q = request.GET.get('q', '').strip()
    when = request.GET.get('when', '').strip()
    if when not in AI_ACTIVITY_DATE_FILTERS:
        when = 'all'
    recording = request.GET.get('recording', '').strip()
    if recording not in ('yes', 'no'):
        recording = 'all'
    length = request.GET.get('length', '').strip()
    if length not in LENGTH_FILTERS:
        length = 'all'
    status = request.GET.get('status', '').strip()
    if status not in ('ended', 'unfinished'):
        status = 'all'
    sort = request.GET.get('sort', '').strip()
    if sort not in SORTS:
        sort = 'recent'
    user_id = request.GET.get('user', '').strip()

    calls = AICall.objects.select_related('user', 'conversation')
    if q:
        # transcript is JSON; searching its text form finds words said on the call.
        calls = calls.filter(
            Q(user__email__icontains=q) | Q(user__username__icontains=q)
            | Q(user__first_name__icontains=q) | Q(user__last_name__icontains=q)
            | Q(caller_name__icontains=q) | Q(transcript__icontains=q)
        )
    if user_id.isdigit():
        calls = calls.filter(user_id=int(user_id))
    now = timezone.localtime()
    if when == 'yesterday':
        start = AI_ACTIVITY_DATE_FILTERS['yesterday'](now)
        calls = calls.filter(started_at__gte=start, started_at__lt=start + timedelta(days=1))
    elif when != 'all':
        calls = calls.filter(started_at__gte=AI_ACTIVITY_DATE_FILTERS[when](now))
    if recording == 'yes':
        calls = calls.filter(audio_bytes__gt=0)
    elif recording == 'no':
        calls = calls.filter(audio_bytes=0)
    if length != 'all':
        calls = calls.filter(LENGTH_FILTERS[length])
    if status == 'ended':
        calls = calls.filter(ended_at__isnull=False)
    elif status == 'unfinished':
        calls = calls.filter(ended_at__isnull=True)

    page = Paginator(calls.order_by(SORTS[sort]), PAGE_SIZE).get_page(request.GET.get('page'))
    rows = []
    for call in page.object_list:
        rows.append({
            'call': call,
            'name': _display_name(call.user),
            'duration': duration_text(call.duration_seconds),
            'preview': ai_calls._preview(call),
            'has_audio': bool(call.audio) and call.audio_bytes > 0,
        })

    today_start = AI_ACTIVITY_DATE_FILTERS['today'](now)
    totals = AICall.objects.aggregate(
        calls=Count('id'), seconds=Sum('duration_seconds'),
        callers=Count('user', distinct=True),
        recorded=Count('id', filter=Q(audio_bytes__gt=0)),
        today=Count('id', filter=Q(started_at__gte=today_start)),
    )
    params = request.GET.copy()
    params.pop('page', None)
    filter_user = None
    if user_id.isdigit():
        first = calls.first()
        filter_user = _display_name(first.user) if first else None
    context = {
        'active': 'calls',
        'rows': rows, 'page': page, 'match_count': page.paginator.count,
        'q': q, 'when': when, 'recording': recording, 'length': length,
        'status': status, 'sort': sort, 'user_id': user_id if user_id.isdigit() else '',
        'filter_user': filter_user, 'query_string': params.urlencode(),
        'stats': {
            'calls': totals['calls'] or 0,
            'today': totals['today'] or 0,
            'talk_time': duration_text(totals['seconds'] or 0),
            'callers': totals['callers'] or 0,
            'recorded': totals['recorded'] or 0,
        },
    }
    return render(request, 'dashboard/calls.html', context)


@dashboard_staff_required
def dashboard_call_detail(request, pk):
    call = get_object_or_404(AICall.objects.select_related('user', 'conversation'), pk=pk)
    you = call.caller_name or _display_name(call.user)
    turns = []
    for turn in call.transcript or []:
        turns.append({
            'ai': turn.get('who') == 'ai',
            'who': None if turn.get('who') == 'ai' else you,
            'text': turn.get('text', ''),
            'time': ai_calls._clock(turn.get('at'), '%I:%M:%S %p'),
            'interrupted': bool(turn.get('interrupted')),
        })
    user_calls = AICall.objects.filter(user=call.user).aggregate(n=Count('id'), seconds=Sum('duration_seconds'))
    context = {
        'active': 'calls',
        'call': call, 'name': _display_name(call.user), 'turns': turns,
        'duration': duration_text(call.duration_seconds),
        'has_audio': bool(call.audio) and call.audio_bytes > 0,
        'audio_mb': round((call.audio_bytes or 0) / (1024 * 1024), 2),
        'user_call_count': user_calls['n'] or 0,
        'user_talk_time': duration_text(user_calls['seconds'] or 0),
    }
    return render(request, 'dashboard/call_detail.html', context)


@dashboard_staff_required
def dashboard_call_audio(request, pk):
    return ai_calls.audio_response(request, get_object_or_404(AICall, pk=pk))


@dashboard_staff_required
def dashboard_call_download(request, pk, file_format):
    if file_format not in ('txt', 'pdf'):
        return redirect('dashboard_call_detail', pk=pk)
    return ai_calls.download_response(get_object_or_404(AICall, pk=pk), file_format)


@dashboard_staff_required
def dashboard_call_delete(request, pk):
    call = get_object_or_404(AICall, pk=pk)
    if request.method == 'POST':
        ai_calls._remove_audio(call)
        call.delete()
        messages.success(request, 'The call, its transcript and its recording were deleted.')
        return redirect('dashboard_calls')
    return redirect('dashboard_call_detail', pk=pk)
