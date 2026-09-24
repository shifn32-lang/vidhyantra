"""Live web lookups for questions whose answer changes over time.

The model's own training data has a fixed cutoff, so anything time-sensitive
("latest", "today's price", "who won", "current rate") is answered from stale
memory unless real results are fetched first. This module supplies that
grounding text; ai_chat.stream_chat treats it exactly like the other
retrieved_context sources (see views.ai_chat_send).

Deliberately cheap and optional:

* ``needs_search`` is a narrow, explicit-signal detector rather than "search
  everything" — a search costs a network round trip on the critical path of a
  chat reply, and most turns (rewrites, code, maths, chat, company questions
  that company_knowledge.py already grounds) genuinely don't need one.
* Results are cached, so a repeated or rephrased-but-identical question in the
  same window costs nothing.
* Every failure path returns ``None`` instead of raising, so a search outage
  degrades to a normal (ungrounded) answer rather than breaking the chat.
"""

import hashlib
import logging
import re

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Kept short on purpose: this runs before the model is even called, so it is
# added latency on every searched turn. A slow search is worse than no search.
SEARCH_TIMEOUT_SECONDS = 4
MAX_RESULTS = 4
# Enough for the model to extract a fact and cite a source, short enough that
# five of them don't crowd out the actual conversation in the context window.
MAX_SNIPPET_CHARS = 320
CACHE_SECONDS = 900


# Explicit "go and look this up" phrasing, including the Hindi/Hinglish forms
# real users type here (see the AIReport analysis — a large share of this
# audience writes "aaj ka", "abhi", "latest kya hai" rather than English).
_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(?:search|google|look\s+up|lookup|find\s+out|web\s+se|internet\s+se|"
    r"search\s+kar(?:o|ke|na)?|dhundo|dhoondo|pata\s+karo|check\s+karo)\b",
    re.IGNORECASE,
)

# Time-sensitive wording — the answer depends on when it's asked.
_FRESHNESS_RE = re.compile(
    r"\b(?:today|todays|today's|tonight|tomorrow|yesterday|now|currently|current|"
    r"latest|newest|recent|recently|this\s+(?:week|month|year|season)|"
    r"last\s+(?:week|month|night)|so\s+far|as\s+of|up\s?to\s?date|updated|"
    r"live|ongoing|upcoming|right\s+now|these\s+days|nowadays|"
    r"aaj|aaj\s?kal|abhi|abhi\s+ka|taaza|taza|naya|nayi|latest\s+kya|"
    r"kya\s+chal\s+raha|chal\s+raha\s+hai)\b",
    re.IGNORECASE,
)

# Topics that are almost always "what is it right now" questions.
_VOLATILE_TOPIC_RE = re.compile(
    r"\b(?:news|headline|headlines|breaking|weather|forecast|temperature|"
    r"score|scores|scorecard|match|result|results|election|poll|"
    r"stock|share\s+price|sensex|nifty|market|ipo|crypto|bitcoin|ethereum|"
    r"gold\s+rate|silver\s+rate|petrol\s+price|diesel\s+price|fuel\s+price|"
    r"exchange\s+rate|conversion\s+rate|repo\s+rate|interest\s+rate|inflation|"
    r"release\s+date|launched|launch\s+date|announced|"
    r"who\s+won|who\s+is\s+the\s+(?:current|new)|"
    r"bhav|daam|rate\s+kya)\b",
    re.IGNORECASE,
)

# "Status of X" is a live-state question even without a freshness word in it.
# AIReport #43 ("Status of Shree cement plant in meghalaya") answered itself
# with "I don't have current operational status data" — the model knew it
# needed live information and no search had been run. Kept narrow: it must be
# the *status of* something, not the bare word "status", which shows up in
# "WhatsApp status", "order status" and every Hinglish sentence about a
# status update.
_LIVE_STATE_RE = re.compile(
    r"\b(?:status|update|progress|situation|condition|standing|position)\s+(?:of|on|about)\b|"
    r"\bis\s+.{1,40}\b(?:still|currently)\s+(?:open|closed|running|operational|working|available)\b|"
    r"\b(?:operational|running|shut\s?down|closed\s+down|under\s+construction)\b|"
    r"\bkya\s+haal\b|\bhaal\s+chaal\b",
    re.IGNORECASE,
)

# A bare year that isn't in the model's reliable range still signals "current
# events" more often than not ("IPL 2026 schedule", "budget 2026 highlights").
_RECENT_YEAR_RE = re.compile(r"\b(?:20[2-9]\d)\b")

# Never search for these, even when the wording above matches: they are either
# already grounded elsewhere, or a search actively wastes time.
_NO_SEARCH_RE = re.compile(
    r"\b(?:rephrase|rewrite|reword|paraphrase|translate|summari[sz]e|proofread|"
    r"code|coding|program|function|debug|traceback|syntax|"
    r"generate\s+(?:an?\s+)?image|create\s+(?:an?\s+)?image|draw|poster|logo|"
    # The user's own account/order state is private to this app — a public web
    # search cannot answer it and would only add a round trip to the reply.
    r"my\s+(?:order|account|subscription|payment|invoice|booking|ticket)|"
    r"order\s+status|track\s+my)\b",
    re.IGNORECASE,
)


# A downloadable reference document (see views._ai_generated_file_spec) is
# explicitly meant to be an accurate, citable artifact — unlike a throwaway
# chat reply, "close enough" from the model's training cutoff isn't good
# enough, and there's no freshness word ("latest", "current") to trigger the
# narrower needs_search() above for something like "PDF of every country's
# population". This is deliberately broader: multi-entry, numeric, or
# ranked real-world data is exactly what a model most confidently hallucinates
# (AIReport: a population-by-country PDF came back with fabricated figures).
_REFERENCE_DATA_RE = re.compile(
    r"\b(?:population|demographics|census|gdp|economy|economies|literacy\s+rate|"
    r"life\s+expectancy|statistics|stats|figures|data(?:set)?|ranking|ranked|"
    r"rank(?:ings)?|top\s+\d+|comparison|compare|capital(?:s)?|currency|"
    r"currencies|exchange\s+rate|area\s+(?:of|in)|coordinates|time\s?zone|"
    r"all\s+countries|every\s+country|list\s+of\s+countries|country\s+list|"
    r"world\s+record|records?)\b",
    re.IGNORECASE,
)


def needs_search_for_document(text):
    """Broader than needs_search() — used only when generating a real,
    downloadable file (see views._ai_generated_file_spec), where getting the
    facts right matters enough to warrant a search even without an explicit
    freshness cue in the wording."""
    text = (text or '').strip()
    if len(text) < 3 or _NO_SEARCH_RE.search(text):
        return False
    return bool(_REFERENCE_DATA_RE.search(text)) or needs_search(text)


def needs_search(text):
    """True when a message's answer plausibly depends on current information.

    Conservative by design — a false positive costs a network round trip on
    every such reply, and a false negative merely means the model answers from
    its own knowledge, exactly as it did before this module existed.
    """
    text = (text or '').strip()
    if len(text) < 3:
        return False
    if _NO_SEARCH_RE.search(text):
        return False
    if _EXPLICIT_SEARCH_RE.search(text):
        return True
    if _FRESHNESS_RE.search(text):
        return True
    if _VOLATILE_TOPIC_RE.search(text):
        return True
    if _LIVE_STATE_RE.search(text):
        return True
    # A year alone is only a signal alongside a real question, not inside a
    # date the user is simply quoting back ("my invoice dated 2026-03-01").
    return bool(_RECENT_YEAR_RE.search(text) and '?' in text)


def search(query, max_results=MAX_RESULTS):
    """Return ``[{title, url, snippet}, ...]``, or ``[]`` on any failure.

    Imports ddgs lazily, the same way doc_extract.py imports fitz — a search
    backend that is missing or broken must degrade to an ungrounded answer,
    never take down the chat endpoint.
    """
    query = (query or '').strip()
    if not query:
        return []

    # hashlib, not hash(): Python randomises string hashing per process, so
    # built-in hash() would give every gunicorn worker a different key for the
    # same query (cache misses, repeated network calls) and reset the whole
    # cache on restart — and a collision would hand back another query's
    # results as if they were this one's.
    digest = hashlib.sha256(query.lower().encode('utf-8')).hexdigest()[:32]
    cache_key = f'websearch:{digest}:{max_results}'
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    from myapp.provider_keys import get_key
    tavily_key = get_key('TAVILY_API_KEY').strip()
    if tavily_key:
        try:
            response = requests.post(
                'https://api.tavily.com/search',
                json={
                    'api_key': tavily_key,
                    'query': query,
                    'search_depth': 'basic',
                    'max_results': max_results,
                    'include_answer': False,
                    'include_raw_content': False,
                },
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            raw = response.json().get('results', [])
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.warning('Tavily search failed for %r: %s', query[:80], exc)
            return []
    else:
        # Keep local/test environments usable when no Tavily key is supplied.
        try:
            from ddgs import DDGS
        except ImportError:
            logger.warning('Web search unavailable: no Tavily key and ddgs is not installed')
            return []
        try:
            raw = DDGS(timeout=SEARCH_TIMEOUT_SECONDS).text(query, max_results=max_results)
        except Exception as exc:
            logger.warning('Web search failed for %r: %s', query[:80], exc)
            return []

    results = []
    for item in raw or []:
        title = (item.get('title') or '').strip()
        url = (item.get('url') or item.get('href') or '').strip()
        snippet = ' '.join(
            (item.get('content') or item.get('body') or '').split()
        )[:MAX_SNIPPET_CHARS]
        if title and url:
            results.append({'title': title, 'url': url, 'snippet': snippet})

    cache.set(cache_key, results, CACHE_SECONDS)
    return results


def build_context(query, max_results=MAX_RESULTS):
    """Format live results as grounding text, or return None when there are none.

    The caller passes this straight to ai_chat.stream_chat as
    ``retrieved_context`` with ``retrieved_source='web_search'``.
    """
    results = search(query, max_results=max_results)
    if not results:
        return None

    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result['title']}\n{result['url']}\n{result['snippet']}")
    return (
        'LIVE WEB RESULTS (fetched just now for this question):\n\n'
        + '\n\n'.join(lines)
        + '\n\nUse these for any fact that depends on current information, and '
        'link a source you actually used with its exact URL above. If they do '
        "not actually answer the question, say so rather than guessing — and "
        'never invent a result, URL, date, or figure that is not shown here.'
    )
