"""Deep research: the plan, reading pages safely, the sources, the cited report
and the two endpoints (plan and run). Nothing here touches the network or a
real model — searches, pages and the model are replaced with canned answers."""
import io
import json
import zipfile
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from myapp import ai_chat, deep_research as dr, views
from myapp.models import AIBlock, AIConversation, AIGeneratedFile, AIMessage, StoreProfile


def public_dns(addresses=('93.184.216.34',)):
    """getaddrinfo that resolves every name to the given public address."""
    return patch('myapp.deep_research.socket.getaddrinfo', return_value=[
        (2, 1, 6, '', (address, 0)) for address in addresses
    ])


def page_response(body, status=200, content_type='text/html; charset=utf-8', headers=None):
    response = MagicMock()
    response.status_code = status
    response.headers = {'Content-Type': content_type, **(headers or {})}
    response.encoding = 'utf-8'
    raw = body.encode('utf-8') if isinstance(body, str) else body
    response.iter_content = lambda size: iter([raw[i:i + size] for i in range(0, len(raw), size)] or [b''])
    return response


ARTICLE = (
    '<html><head><title>T</title><script>var tracking = 1;</script></head><body>'
    '<nav><li>Home navigation entry that is certainly long enough to be kept otherwise</li></nav>'
    '<article><h1>Solar subsidy changes</h1>'
    '<p>The government raised the rooftop solar subsidy for homes in the latest scheme update this year.</p>'
    '<ul><li>Households installing up to three kilowatts receive the highest central subsidy rate.</li></ul>'
    '<p>The government raised the rooftop solar subsidy for homes in the latest scheme update this year.</p>'
    '<p>Short</p></article>'
    '<footer><p>Footer text that is also long enough to be kept if the footer was not removed first.</p></footer>'
    '</body></html>'
)


class PlanTests(SimpleTestCase):
    def test_steps_are_cleaned_deduplicated_and_capped(self):
        raw = [{'text': '**Review** the data', 'query': '"solar subsidy"'}, 'Review the data', {'text': ''}, 5, None]
        raw += [{'text': f'Step number {i}'} for i in range(20)]
        steps = dr.clean_steps(raw)
        self.assertEqual(steps[0], {'text': 'Review the data', 'query': 'solar subsidy'})
        self.assertEqual(len(steps), dr.MAX_STEPS)
        self.assertEqual(len({step['text'].lower() for step in steps}), len(steps))

    def test_a_typed_step_gets_a_search_of_its_own(self):
        query = dr.derive_query('Collect official Anthropic documentation and Claude product pages.', 'Claude AI')
        self.assertIn('Anthropic documentation', query)
        self.assertLessEqual(len(query), dr.MAX_QUERY_CHARS)
        self.assertFalse(query.lower().startswith('collect'))

    def test_fallback_plan_is_usable_without_a_model(self):
        plan = dr.fallback_plan('Impact of GST on small businesses?')
        self.assertGreaterEqual(len(plan['steps']), dr.MIN_STEPS)
        self.assertTrue(all(step['query'] for step in plan['steps']))
        shown = dr.plan_for_page(plan)
        self.assertTrue(shown['steps'][-1]['final'])
        self.assertEqual(shown['steps'][-1]['query'], '')
        self.assertFalse(any(step['final'] for step in shown['steps'][:-1]))

    def test_model_plan_is_used_when_it_is_valid(self):
        reply = {
            'title': 'Research solar subsidies',
            'steps': [{'text': f'Review part {i}', 'query': f'solar part {i}'} for i in range(4)] + [{'text': 'No query here'}],
            'final': 'Summarise it.',
        }
        with patch('myapp.deep_research.ai_chat.complete_json', return_value=reply):
            plan = dr.make_plan('solar subsidies in India', model_key='quick')
        self.assertEqual(plan['title'], 'Research solar subsidies')
        self.assertEqual(len(plan['steps']), 5)
        self.assertTrue(plan['steps'][-1]['query'])         # derived for the step that had none
        self.assertNotIn('fallback', plan)

    def test_plan_falls_back_when_the_model_fails_or_says_too_little(self):
        with patch('myapp.deep_research.ai_chat.complete_json', side_effect=RuntimeError('down')):
            self.assertTrue(dr.make_plan('solar subsidies', model_key='quick')['fallback'])
        with patch('myapp.deep_research.ai_chat.complete_json', return_value={'title': 'x', 'steps': [{'text': 'Only one'}]}):
            self.assertTrue(dr.make_plan('solar subsidies', model_key='quick')['fallback'])

    def test_confirmed_plan_from_the_page_is_cleaned(self):
        raw = [
            {'text': 'Find the official numbers', 'query': ''},
            {'text': '<b>Compare</b> providers', 'query': 'provider comparison'},
            {'text': 'Write it up my way', 'final': True},
            'not a dict',
        ]
        plan = dr.plan_from_page('  My title  ', raw, 'solar subsidies in India')
        self.assertEqual(plan['title'], 'My title')
        self.assertEqual(plan['final'], 'Write it up my way')
        self.assertEqual(len(plan['steps']), 2)
        self.assertTrue(plan['steps'][0]['query'])
        empty = dr.plan_from_page('', [], 'solar subsidies in India')
        self.assertGreaterEqual(len(empty['steps']), dr.MIN_STEPS)
        self.assertEqual(empty['final'], dr.DEFAULT_FINAL_STEP)

    def test_chat_context_keeps_the_last_turns_short(self):
        turns = [('user', 'a' * 900), ('assistant', 'b'), ('user', ''), ('assistant', 'c')] + [('user', f'q{i}') for i in range(6)]
        lines = dr.chat_context(turns).split('\n')
        self.assertEqual(len(lines), dr.CONTEXT_TURNS)
        self.assertTrue(all(len(line) <= dr.CONTEXT_TURN_CHARS + 12 for line in lines))


class JsonAnswerTests(SimpleTestCase):
    def test_json_is_found_inside_fences_and_chatter(self):
        self.assertEqual(ai_chat._json_object_from('{"a": 1}'), {'a': 1})
        self.assertEqual(ai_chat._json_object_from('```json\n{"a": 1}\n```'), {'a': 1})
        self.assertEqual(ai_chat._json_object_from('Here you go: {"a": {"b": 2}} done'), {'a': {'b': 2}})
        for bad in ('', 'no json here', '[1, 2]', '{broken'):
            with self.assertRaises(ValueError):
                ai_chat._json_object_from(bad)


class BackendChoiceTests(TestCase):
    """complete_json must send a turn to the same backend and key stream_chat does."""

    def test_same_backend_and_key_as_a_chat_turn(self):
        for model_key, identity in (
            ('quick', None), ('quick', ai_chat.CHATGPT_56_MODEL_KEY), (ai_chat.SOL_MODEL_KEY, None),
            (ai_chat.TERRA_MODEL_KEY, None), ('ultra', None),
        ):
            with self.subTest(model=model_key, identity=identity):
                seen = {}

                def fake_client(setting):
                    seen['setting'] = setting
                    return object()

                def fake_stream(client, kwargs):
                    seen['model'] = kwargs['model']
                    yield 'Hello there, this is a perfectly ordinary reply.'

                with patch('myapp.ai_chat._get_client', side_effect=fake_client), \
                        patch('myapp.ai_chat._stream_content', side_effect=fake_stream):
                    list(ai_chat.stream_chat(
                        [{'role': 'user', 'content': 'hello'}], model_key=model_key, identity_model_key=identity,
                    ))
                chat_turn = dict(seen)
                seen.clear()
                with patch('myapp.ai_chat._get_client', side_effect=fake_client), \
                        patch('myapp.ai_chat._stream_content', side_effect=lambda c, k: iter(['{"ok": true}']) if seen.update(model=k['model']) is None else None):
                    ai_chat.complete_json('Reply with JSON.', 'hi', model_key=model_key, identity_model_key=identity)
                self.assertEqual(seen['setting'], chat_turn['setting'])
                self.assertEqual(seen['model'], chat_turn['model'])


class PageSafetyTests(SimpleTestCase):
    def test_only_public_addresses_are_allowed(self):
        for host in ('localhost', '127.0.0.1', '10.1.2.3', '169.254.169.254', '192.168.0.9', '172.16.5.5',
                     '[::1]', 'printer.local', 'db.internal', '', None):
            with self.subTest(host=host):
                self.assertFalse(dr.is_public_host(host))
        with public_dns():
            self.assertTrue(dr.is_public_host('example.com'))
        with public_dns(('93.184.216.34', '10.0.0.8')):       # one private answer is enough to refuse
            self.assertFalse(dr.is_public_host('mixed.example.com'))
        with public_dns(('::ffff:10.0.0.8',)):
            self.assertFalse(dr.is_public_host('mapped.example.com'))
        with patch('myapp.deep_research.socket.getaddrinfo', side_effect=OSError('no such host')):
            self.assertFalse(dr.is_public_host('nowhere.example.com'))

    def test_odd_schemes_and_ports_are_refused(self):
        with public_dns():
            self.assertTrue(dr._page_allowed('https://example.com/a'))
            self.assertFalse(dr._page_allowed('ftp://example.com/a'))
            self.assertFalse(dr._page_allowed('file:///etc/passwd'))
            self.assertFalse(dr._page_allowed('http://example.com:8080/a'))
            self.assertFalse(dr._page_allowed('http://example.com:notaport/a'))

    def test_article_text_is_extracted_without_menus_and_repeats(self):
        text = dr.extract_text(ARTICLE)
        self.assertIn('Solar subsidy changes', text)
        self.assertIn('Households installing up to three kilowatts', text)
        self.assertNotIn('navigation entry', text)
        self.assertNotIn('Footer text', text)
        self.assertNotIn('tracking', text)
        self.assertEqual(text.count('The government raised the rooftop solar subsidy'), 1)
        self.assertLessEqual(len(dr.extract_text(ARTICLE * 1, max_chars=120)), 120)

    def test_a_page_is_read(self):
        with public_dns(), patch('myapp.deep_research.requests.Session.get', return_value=page_response(ARTICLE)):
            self.assertIn('rooftop solar subsidy', dr.fetch_page_text('https://example.com/solar'))

    def test_a_redirect_to_a_private_address_is_not_followed(self):
        hop = page_response('', status=302, headers={'Location': 'http://169.254.169.254/latest/meta-data/'})
        with patch('myapp.deep_research.socket.getaddrinfo', side_effect=lambda host, *a, **k: [
            (2, 1, 6, '', ('93.184.216.34' if host == 'example.com' else host, 0)),
        ]), patch('myapp.deep_research.requests.Session.get', return_value=hop) as get:
            self.assertEqual(dr.fetch_page_text('https://example.com/go'), '')
        self.assertEqual(get.call_count, 1)

    def test_redirects_are_followed_a_few_times_then_dropped(self):
        loop = page_response('', status=301, headers={'Location': '/again'})
        with public_dns(), patch('myapp.deep_research.requests.Session.get', return_value=loop) as get:
            self.assertEqual(dr.fetch_page_text('https://example.com/loop'), '')
        self.assertEqual(get.call_count, dr.PAGE_MAX_REDIRECTS + 1)
        hop = page_response('', status=302, headers={'Location': '/final'})
        with public_dns(), patch('myapp.deep_research.requests.Session.get', side_effect=[hop, page_response(ARTICLE)]):
            self.assertIn('rooftop solar subsidy', dr.fetch_page_text('https://example.com/start'))

    def test_things_that_are_not_pages_are_skipped(self):
        for response in (
            page_response('%PDF-1.4', content_type='application/pdf'),
            page_response('<html></html>', status=404),
            page_response('x', content_type='image/png'),
        ):
            with public_dns(), patch('myapp.deep_research.requests.Session.get', return_value=response):
                self.assertEqual(dr.fetch_page_text('https://example.com/file'), '')
        with public_dns(), patch('myapp.deep_research.requests.Session.get', side_effect=dr.requests.ConnectionError('boom')):
            self.assertEqual(dr.fetch_page_text('https://example.com/down'), '')

    def test_plain_text_pages_are_read_and_a_huge_body_is_cut(self):
        body = 'word ' * 200_000
        with public_dns(), patch('myapp.deep_research.requests.Session.get', return_value=page_response(body, content_type='text/plain')):
            text = dr.fetch_page_text('https://example.com/big.txt', max_chars=300)
        self.assertEqual(len(text), 300)


class SourcePoolTests(SimpleTestCase):
    def test_pool_numbers_sources_and_refuses_repeats(self):
        pool = dr.SourcePool()
        first = pool.add({'title': 'A', 'url': 'https://www.example.com/page/?utm_source=x#top', 'snippet': 's'})
        self.assertEqual(first['n'], 1)
        self.assertEqual(first['host'], 'example.com')
        self.assertIsNone(pool.add({'title': 'A again', 'url': 'https://example.com/page'}))
        self.assertEqual(pool.add({'title': 'B', 'url': 'https://other.org/x'})['n'], 2)

    def test_a_site_is_capped_and_social_pages_are_skipped(self):
        pool = dr.SourcePool()
        for number in range(5):
            pool.add({'title': f'P{number}', 'url': f'https://one.example/{number}'})
        self.assertEqual(len(pool), dr.MAX_PER_DOMAIN)
        for url in ('https://facebook.com/x', 'https://m.facebook.com/x', 'https://x.com/a', 'ftp://a.example/x', 'not a url'):
            self.assertIsNone(pool.add({'title': 'T', 'url': url}))
        self.assertIsNone(pool.add({'title': '', 'url': 'https://two.example/x'}))

    def test_the_pool_has_a_size_limit(self):
        pool = dr.SourcePool(limit=3)
        for number in range(10):
            pool.add({'title': f'T{number}', 'url': f'https://site{number}.example/x'})
        self.assertEqual(len(pool), 3)

    def test_evidence_prefers_a_real_page_over_the_search_extract(self):
        pool = dr.SourcePool()
        source = pool.add({'title': 'A', 'url': 'https://a.example/x', 'snippet': 'short extract'})
        self.assertEqual(pool.evidence(source), 'short extract')
        source['text'] = 'long page text. ' * 60
        self.assertTrue(pool.evidence(source).startswith('long page text'))
        source['text'] = 'tiny'
        self.assertEqual(pool.evidence(source), 'short extract')

    def test_context_numbers_every_source_and_warns_about_untrusted_text(self):
        pool = dr.SourcePool()
        pool.add({'title': 'First', 'url': 'https://a.example/x', 'snippet': 'alpha'})
        pool.add({'title': 'Second', 'url': 'https://b.example/y', 'snippet': 'beta'})
        context = dr.build_context(pool)
        self.assertIn('[1] First (a.example)', context)
        self.assertIn('[2] Second (b.example)', context)
        self.assertIn('https://b.example/y', context)
        self.assertIn('untrusted', context)
        self.assertIsNone(dr.build_context(dr.SourcePool()))


class ResearchRunTests(SimpleTestCase):
    steps = [
        {'text': 'Find official sources', 'query': 'solar official'},
        {'text': 'Collect statistics', 'query': 'solar statistics'},
    ]

    @staticmethod
    def fake_search(query, max_results=4, *, deep=False):
        slug = query.replace(' ', '-')
        return [
            {'title': f'{query} {i}', 'url': f'https://{slug}-{i}.example/page', 'snippet': f'extract {i}'}
            for i in range(1, 5)
        ]

    def test_each_step_searches_and_reads_the_best_new_pages(self):
        pool = dr.SourcePool()
        with patch('myapp.deep_research.web_search.search', side_effect=self.fake_search) as search, \
                patch('myapp.deep_research.fetch_page_text', return_value='page text ' * 80) as fetch:
            events = list(dr.research(self.steps, pool, topic='solar', question='solar power'))
        self.assertEqual(search.call_count, 2)
        self.assertTrue(all(call.kwargs.get('deep') for call in search.call_args_list))
        self.assertEqual(fetch.call_count, 2 * dr.READ_PER_STEP)
        kinds = [(event['t'], event.get('s')) for event in events if event['t'] == 'step']
        self.assertEqual(kinds, [('step', 'active'), ('step', 'done'), ('step', 'active'), ('step', 'done')])
        self.assertEqual(len([event for event in events if event['t'] == 'src']), 8)
        self.assertTrue(pool.items[0]['text'])
        self.assertFalse(pool.items[3]['text'])       # only the best few per step are read in full

    def test_a_failing_search_just_means_fewer_sources(self):
        pool = dr.SourcePool()
        with patch('myapp.deep_research.web_search.search', return_value=[]) as search, \
                patch('myapp.deep_research.fetch_page_text') as fetch:
            events = list(dr.research(self.steps, pool, topic='solar', question='solar power'))
        self.assertEqual(len(pool), 0)
        fetch.assert_not_called()
        self.assertEqual([event['s'] for event in events if event['t'] == 'step'], ['active', 'done', 'active', 'done'])
        self.assertEqual(search.call_count, 3)        # the two steps, then one broad search of the question itself

    def test_one_broad_search_rescues_a_thin_result(self):
        pool = dr.SourcePool()
        answers = iter([[], [], self.fake_search('broad')[:3]])
        with patch('myapp.deep_research.web_search.search', side_effect=lambda *a, **k: next(answers)), \
                patch('myapp.deep_research.fetch_page_text', return_value=''):
            list(dr.research(self.steps, pool, topic='solar', question='solar power'))
        self.assertEqual(len(pool), 3)

    def test_the_time_budget_stops_new_work(self):
        pool = dr.SourcePool()
        with patch('myapp.deep_research.RESEARCH_SECONDS', -1), \
                patch('myapp.deep_research.web_search.search') as search:
            events = list(dr.research(self.steps, pool, topic='solar', question='solar power'))
        search.assert_not_called()
        self.assertEqual(len([event for event in events if event['t'] == 'step']), 4)

    def test_a_page_that_cannot_be_read_falls_back_to_its_extract(self):
        pool = dr.SourcePool()
        with patch('myapp.deep_research.web_search.search', side_effect=self.fake_search), \
                patch('myapp.deep_research.fetch_page_text', side_effect=RuntimeError('boom')):
            list(dr.research(self.steps[:1], pool, topic='solar', question='solar power'))
        self.assertEqual(pool.evidence(pool.items[0]), 'extract 1')


class FinalReportTests(SimpleTestCase):
    def pool(self, count=5):
        pool = dr.SourcePool()
        for number in range(1, count + 1):
            pool.add({'title': f'Title {number} [x]', 'url': f'https://site{number}.example/page(1)', 'snippet': 's'})
        return pool

    REPORT = (
        "Sure! Here is the report:\n\n# Solar subsidies in India\n\n> **Summary:** Subsidies exist [1, 2].\n\n"
        "## Key findings\n\n- Rooftop solar gets a subsidy [3][9].\n- A range such as [10, 20] is not a citation.\n"
        "- A link-style cite [4](https://evil.example/x).\n\n```\ncode [1] stays\n```\n\n"
        "## Sources of growth\n\nGrowth text [5].\n\n## Sources\n\n1. [Fake](https://fake.example)\n\n"
        "## Conclusion\n\nDone [5] [1] .\n"
    )

    def test_citations_are_renumbered_in_order_and_unknown_ones_removed(self):
        markdown, cited = dr.finalize_report(self.REPORT, self.pool(), plan_title='X')
        self.assertTrue(markdown.startswith('# Solar subsidies in India'))      # chatter before the title is gone
        self.assertIn('Subsidies exist [1][2].', markdown)
        self.assertIn('Rooftop solar gets a subsidy [3].', markdown)             # [9] is not a source
        self.assertIn('[10, 20]', markdown)                                      # a range is left alone
        self.assertIn('A link-style cite [4].', markdown)
        self.assertIn('code [1] stays', markdown)                                # code is never touched
        self.assertIn('Growth text [5].', markdown)
        self.assertIn('Done [5] [1].', markdown)                                 # a space before the full stop is tidied
        self.assertEqual([source['n'] for source in cited], [1, 2, 3, 4, 5])

    def test_renumbering_follows_the_order_of_first_appearance(self):
        text = '# T\n\nFirst [4]. Second [2]. Again [4]. Third [1].'
        markdown, cited = dr.finalize_report(text, self.pool(), plan_title='T')
        self.assertIn('First [1]. Second [2]. Again [1]. Third [3].', markdown)
        self.assertEqual([source['title'] for source in cited], ['Title 4 [x]', 'Title 2 [x]', 'Title 1 [x]'])
        self.assertIn('1. [Title 4 (x)](https://site4.example/page%281%29)', markdown)
        self.assertIn('3. [Title 1 (x)](https://site1.example/page%281%29)', markdown)

    def test_a_sources_section_written_by_the_model_is_replaced_by_the_real_one(self):
        markdown, _ = dr.finalize_report(self.REPORT, self.pool(), plan_title='X')
        self.assertNotIn('fake.example', markdown)
        self.assertIn('## Sources of growth', markdown)
        self.assertEqual(markdown.count('\n## Sources\n'), 1)
        self.assertLess(markdown.index('## Conclusion'), markdown.index('\n## Sources\n'))

    def test_a_title_is_added_and_a_whole_document_fence_removed(self):
        markdown, _ = dr.finalize_report('## Overview\n\nPlain text.', self.pool(0), plan_title='Research solar')
        self.assertTrue(markdown.startswith('# Research solar\n'))
        markdown, _ = dr.finalize_report('```markdown\n# Wrapped\n\nBody text.\n```', self.pool(0), plan_title='X')
        self.assertTrue(markdown.startswith('# Wrapped'))

    def test_report_without_citations_lists_what_was_reviewed(self):
        markdown, cited = dr.finalize_report('# T\n\nNo numbers here.', self.pool(3), plan_title='T')
        self.assertEqual(cited, [])
        self.assertIn('## Sources reviewed', markdown)
        self.assertIn('from 3 web sources', markdown)

    def test_report_without_any_sources_says_so(self):
        markdown, cited = dr.finalize_report('# T\n\nBody [1].', self.pool(0), plan_title='T')
        self.assertEqual(cited, [])
        self.assertNotIn('[1]', markdown)
        self.assertNotIn('## Sources', markdown)
        self.assertIn('without live web sources', markdown)

    def test_the_finished_report_makes_real_documents(self):
        markdown, _ = dr.finalize_report(self.REPORT, self.pool(), plan_title='X')
        pdf = views._ai_pdf_bytes(markdown, brand='Vidhyora', created=timezone.now())
        self.assertTrue(pdf.startswith(b'%PDF'))
        from pypdf import PdfReader
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(pdf)).pages)
        self.assertIn('Solar subsidies in India', text)
        self.assertIn('Sources', text)
        self.assertIn('site1.example', text)
        self.assertIn('Summary', text.title() if 'SUMMARY' in text else text)
        word = views._ai_word_document_bytes(markdown, brand='Vidhyora', created=timezone.now())
        with zipfile.ZipFile(io.BytesIO(word)) as archive:
            xml = archive.read('word/document.xml').decode('utf-8')
        self.assertIn('Solar subsidies in India', xml)
        self.assertIn('site1.example', xml)

    def test_a_summary_callout_is_labelled_summary(self):
        from myapp import doc_blocks
        block = next(b for b in doc_blocks.parse_markdown('> **Summary:** Short answer.') if b.kind == 'quote')
        self.assertEqual(block.callout, 'summary')
        self.assertEqual(doc_blocks.CALLOUTS['summary'][2], 'Summary')

    def test_file_slugs(self):
        self.assertEqual(dr.file_slug('Solar subsidies in India: 2026 — what next?'), 'solar-subsidies-in-india-2026-what-next')
        self.assertEqual(dr.file_slug('हिंदी'), '')
        self.assertLessEqual(len(dr.file_slug('word ' * 40)), 50)


class InstructionTests(SimpleTestCase):
    def test_instruction_asks_for_cited_structure(self):
        text = dr.report_instruction(has_sources=True)
        for needle in ('Key findings', 'Summary', 'square brackets', 'never put a specific statistic', 'Sources or References'):
            self.assertIn(needle, text)

    def test_without_sources_the_report_must_say_so(self):
        text = dr.report_instruction(has_sources=False)
        self.assertIn('could not be reached', text)
        self.assertIn('no citation numbers', text)

    def test_brief_lists_the_plan_and_the_chat_context(self):
        plan = {'steps': [{'text': 'One'}, {'text': 'Two'}]}
        brief = dr.research_brief('Why is the sky blue?', plan, 'User: hi')
        self.assertIn('Research question: Why is the sky blue?', brief)
        self.assertIn('1. One', brief)
        self.assertIn('User: hi', brief)


GOOD_REPORT = (
    "# Rooftop solar in India\n\n> **Summary:** Rooftop solar is supported by a central subsidy [2].\n\n"
    "## Key findings\n\n- Households get a subsidy for small systems [2].\n- Costs fell over the last decade [1].\n\n"
    "## Costs and payback\n\nPayback depends on the tariff and the system size, and sources give different figures [1][3]. "
    "Background: a rooftop system converts sunlight into household electricity during the day, which offsets grid "
    "purchases and reduces the monthly bill for the owner over the system's life.\n\n"
    "| Item | Detail |\n|---|---|\n| Subsidy | Central scheme [2] |\n| Payback | Varies [1][3] |\n\n"
    "## Risks, limitations and open questions\n\nSubsidy rules change often, so check the current scheme before buying.\n\n"
    "## Conclusion and recommendations\n\nCompare quotes, check the scheme and plan for the payback period.\n\n"
    "## Sources\n\n1. [Invented](https://invented.example)\n"
)


def canned_search(query, max_results=4, *, deep=False):
    slug = ''.join(ch if ch.isalnum() else '-' for ch in query)
    return [{'title': f'Source for {query}', 'url': f'https://{slug}.example/page', 'snippet': 'An extract about solar.'}]


class ResearchEndpointTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create_user(username='researcher@example.com', email='researcher@example.com', password='x-pass-12345')
        self.client.force_login(self.user)
        self.plan = {
            'title': 'Research rooftop solar',
            'steps': [{'text': 'Find official sources', 'query': 'solar official'},
                      {'text': 'Collect costs', 'query': 'solar costs'},
                      {'text': 'Compare payback', 'query': 'solar payback'}],
            'final': 'Summarise.',
        }

    def post(self, name, body, **extra):
        return self.client.post(f'/AI/api/research/{name}/', json.dumps(body), content_type='application/json', **extra)

    def events(self, response):
        raw = b''.join(response.streaming_content).decode('utf-8')
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def run_report(self, body=None, report=GOOD_REPORT, **extra):
        body = body or {'message': 'rooftop solar in India', 'steps': dr.plan_for_page(self.plan)['steps'], 'title': 'Research rooftop solar'}
        chunks = [report[i:i + 60] for i in range(0, len(report), 60)]
        with patch('myapp.deep_research.web_search.search', side_effect=canned_search), \
                patch('myapp.deep_research.fetch_page_text', return_value='page text about solar ' * 40), \
                patch('myapp.views.ai_chat.stream_chat', return_value=iter(chunks)) as stream:
            response = self.post('run', body, **extra)
            events = self.events(response) if response.status_code == 200 else None
        return response, events, stream

    def make_premium(self):
        StoreProfile.objects.update_or_create(user=self.user, defaults={'ai_subscription_until': timezone.now() + timedelta(days=30)})

    # ── plan ──

    def test_a_guest_is_asked_to_log_in(self):
        self.client.logout()
        for name in ('plan', 'run'):
            response = self.post(name, {'message': 'rooftop solar'})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()['status'], 'login_required')

    def test_get_is_not_allowed(self):
        self.assertEqual(self.client.get('/AI/api/research/plan/').status_code, 405)

    def test_plan_returns_the_steps_and_a_closing_step(self):
        with patch('myapp.deep_research.ai_chat.complete_json', return_value={
            'title': 'Research rooftop solar', 'steps': self.plan['steps'], 'final': 'Summarise it.',
        }):
            response = self.post('plan', {'message': 'rooftop solar in India'})
        body = response.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['title'], 'Research rooftop solar')
        self.assertEqual(len(body['steps']), 4)
        self.assertTrue(body['steps'][-1]['final'])
        self.assertFalse(body['fallback'])
        self.assertEqual(AIMessage.objects.count(), 0)             # nothing is saved until it is started

    def test_plan_still_works_when_the_model_is_down(self):
        with patch('myapp.deep_research.ai_chat.complete_json', side_effect=RuntimeError('down')):
            response = self.post('plan', {'message': 'rooftop solar in India'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['fallback'])
        self.assertGreaterEqual(len(response.json()['steps']), 4)

    def test_plan_checks_the_question_the_block_and_the_daily_limit(self):
        self.assertEqual(self.post('plan', {'message': 'hi'}).status_code, 400)
        self.assertEqual(self.post('plan', 'not an object').status_code, 400)
        AIBlock.objects.create(user=self.user, reason='spam')
        self.assertEqual(self.post('plan', {'message': 'rooftop solar in India'}).status_code, 403)
        AIBlock.objects.all().delete()
        for number in range(views.AI_RESEARCH_DAILY_LIMIT):
            AIGeneratedFile.objects.create(user=self.user, file_name=f'deep-research-x{number}.md', content='# x')
        response = self.post('plan', {'message': 'rooftop solar in India'})
        self.assertEqual(response.status_code, 429)
        self.assertIn('premium access raises it', response.json()['detail'])
        self.make_premium()
        with patch('myapp.deep_research.ai_chat.complete_json', side_effect=RuntimeError('down')):
            self.assertEqual(self.post('plan', {'message': 'rooftop solar in India'}).status_code, 200)

    def test_plan_is_rate_limited(self):
        with patch('myapp.deep_research.ai_chat.complete_json', side_effect=RuntimeError('down')):
            codes = [self.post('plan', {'message': 'rooftop solar in India'}).status_code for _ in range(views.AI_RESEARCH_PLAN_LIMIT + 1)]
        self.assertEqual(codes[:-1], [200] * views.AI_RESEARCH_PLAN_LIMIT)
        self.assertEqual(codes[-1], 429)

    # ── run ──

    def test_a_run_streams_progress_then_the_finished_report(self):
        response, events, stream = self.run_report()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/x-ndjson; charset=utf-8')
        kinds = [event['t'] for event in events]
        self.assertEqual(kinds[:2], ['plan', 'step'])
        self.assertEqual(len(events[0]['steps']), 4)                  # three steps and the closing one, as the server will run them
        self.assertTrue(events[0]['steps'][-1]['final'])
        self.assertIn('src', kinds)
        self.assertIn('write', kinds)
        self.assertIn('d', kinds)
        self.assertEqual(kinds[-1], 'final')
        self.assertLess(kinds.index('src'), kinds.index('write'))
        self.assertLess(kinds.index('write'), kinds.index('d'))
        self.assertEqual(len([e for e in events if e['t'] == 'step' and e['s'] == 'done']), 3)
        final = events[-1]
        self.assertIn('Rooftop solar in India', final['md'])
        self.assertNotIn('invented.example', final['md'])           # the model's own Sources list is replaced
        self.assertIn('## Sources', final['md'])
        self.assertTrue(final['md'].rstrip().splitlines()[-1].startswith('**Download this report:**'))
        self.assertEqual([item['kind'] for item in final['files']], ['pdf', 'docx', 'md'])
        self.assertEqual(final['stats']['reviewed'], 3)
        self.assertEqual(final['stats']['cited'], len(final['sources']))
        self.assertTrue(response['X-Conversation-Id'])
        self.assertTrue(response['X-User-Message-Id'])

    def test_the_conversation_and_files_are_saved(self):
        response, events, _ = self.run_report()
        final = events[-1]
        conversation = AIConversation.objects.get(pk=int(response['X-Conversation-Id']))
        self.assertEqual(conversation.user, self.user)
        roles = list(conversation.messages.order_by('pk').values_list('role', flat=True))
        self.assertEqual(roles, ['user', 'assistant'])
        assistant = conversation.messages.get(role='assistant')
        self.assertEqual(assistant.content, final['md'])
        self.assertEqual(assistant.pk, final['message_id'])
        files = list(AIGeneratedFile.objects.filter(user=self.user))
        self.assertEqual(sorted(f.file_name.rsplit('.', 1)[1] for f in files), ['docx', 'md', 'pdf'])
        self.assertTrue(all(f.file_name.startswith('deep-research-rooftop-solar-in-india') for f in files))
        self.assertEqual(len({f.content for f in files}), 1)
        self.assertNotIn('Download this report', files[0].content)
        # the links in the chat message are the real downloads
        for item in final['files']:
            download = self.client.get(item['url'].replace('http://testserver', ''))
            self.assertEqual(download.status_code, 200, item['url'])
        pdf = self.client.get(final['files'][0]['url'].replace('http://testserver', ''))
        self.assertTrue(b''.join(pdf.streaming_content if pdf.streaming else [pdf.content]).startswith(b'%PDF'))
        word = self.client.get(final['files'][1]['url'].replace('http://testserver', ''))
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(word.content)))
        markdown = self.client.get(final['files'][2]['url'].replace('http://testserver', ''))
        self.assertIn('Rooftop solar in India', markdown.content.decode('utf-8'))

    def test_the_report_is_written_from_the_numbered_sources(self):
        _, _, stream = self.run_report()
        kwargs = stream.call_args.kwargs
        self.assertIn('[1] Source for solar official', kwargs['retrieved_context'])
        self.assertEqual(kwargs['retrieved_source'], 'web_search')
        self.assertIn('DEEP RESEARCH REPORT', kwargs['document_instruction'])
        self.assertEqual(kwargs['max_tokens'], dr.REPORT_MAX_TOKENS)
        self.assertIn('Research question: rooftop solar in India', stream.call_args.args[0][0]['content'])

    def test_a_free_account_gets_quick_and_counts_one_message(self):
        before = StoreProfile.objects.get_or_create(user=self.user)[0].ai_free_messages_used
        _, _, stream = self.run_report(body={
            'message': 'rooftop solar in India', 'model': 'sol', 'steps': dr.plan_for_page(self.plan)['steps'],
        })
        self.assertEqual(stream.call_args.kwargs['model_key'], 'quick')
        self.assertIsNone(stream.call_args.kwargs['identity_model_key'])
        self.assertEqual(StoreProfile.objects.get(user=self.user).ai_free_messages_used, before + 1)
        self.assertEqual(AIMessage.objects.get(role='assistant').model_key, 'quick')

    def test_a_subscriber_keeps_the_chosen_model(self):
        self.make_premium()
        steps = dr.plan_for_page(self.plan)['steps']
        _, _, stream = self.run_report(body={'message': 'rooftop solar in India', 'model': ai_chat.TERRA_MODEL_KEY, 'steps': steps})
        self.assertEqual(stream.call_args.kwargs['model_key'], ai_chat.TERRA_MODEL_KEY)
        cache.clear()
        _, _, stream = self.run_report(body={'message': 'rooftop solar in India', 'model': ai_chat.CHATGPT_56_MODEL_KEY, 'steps': steps})
        self.assertEqual(stream.call_args.kwargs['model_key'], 'quick')
        self.assertEqual(stream.call_args.kwargs['identity_model_key'], ai_chat.CHATGPT_56_MODEL_KEY)
        cache.clear()
        _, _, stream = self.run_report(body={'message': 'rooftop solar in India', 'model': 'code', 'steps': steps})
        self.assertEqual(stream.call_args.kwargs['model_key'], 'quick')            # a report is not a coding task

    def test_a_vendor_name_never_reaches_the_report(self):
        report = GOOD_REPORT.replace('Costs fell over', 'I was trained by NVIDIA. Costs fell over')
        _, events, _ = self.run_report(report=report)
        final = events[-1]['md']
        shown = ''.join(event['x'] for event in events if event['t'] == 'd')
        self.assertNotIn('NVIDIA', final)
        self.assertNotIn('NVIDIA', shown)

    def test_a_report_that_is_too_short_is_not_saved(self):
        _, events, _ = self.run_report(report='# Tiny\n\nToo short.')
        self.assertEqual(events[-1]['t'], 'error')
        self.assertFalse(AIGeneratedFile.objects.exists())
        self.assertFalse(AIMessage.objects.filter(role='assistant').exists())
        self.assertIsNone(cache.get(f'ai_research_active:{self.user.pk}'))        # the lock is released

    def test_a_model_failure_is_reported_in_plain_words_and_nothing_is_saved(self):
        def broken(*args, **kwargs):
            yield 'A start'
            raise ai_chat.ModelDisabledError('off')
        with patch('myapp.deep_research.web_search.search', side_effect=canned_search), \
                patch('myapp.deep_research.fetch_page_text', return_value=''), \
                patch('myapp.views.ai_chat.stream_chat', side_effect=broken):
            response = self.post('run', {'message': 'rooftop solar in India', 'steps': dr.plan_for_page(self.plan)['steps']})
            events = self.events(response)
        self.assertEqual(events[-1]['t'], 'error')
        self.assertNotIn('Traceback', events[-1]['msg'])
        self.assertFalse(AIGeneratedFile.objects.exists())
        self.assertEqual(AIMessage.objects.filter(role='assistant').count(), 0)

    def test_a_report_can_be_written_even_when_no_source_is_found(self):
        with patch('myapp.deep_research.web_search.search', return_value=[]), \
                patch('myapp.views.ai_chat.stream_chat', return_value=iter([GOOD_REPORT])) as stream:
            response = self.post('run', {'message': 'rooftop solar in India', 'steps': dr.plan_for_page(self.plan)['steps']})
            events = self.events(response)
        self.assertEqual(events[-1]['t'], 'final')
        self.assertIsNone(stream.call_args.kwargs['retrieved_context'])
        self.assertIn('could not be reached', stream.call_args.kwargs['document_instruction'])
        self.assertIn('without live web sources', events[-1]['md'])
        self.assertEqual(events[-1]['sources'], [])

    def test_one_report_at_a_time_per_account(self):
        cache.set(f'ai_research_active:{self.user.pk}', True, 60)
        response = self.post('run', {'message': 'rooftop solar in India'})
        self.assertEqual(response.status_code, 409)
        self.assertFalse(AIMessage.objects.exists())

    def test_a_busy_server_says_so_and_saves_nothing(self):
        slots = [views._AI_RESEARCH_SLOTS.acquire(blocking=False) for _ in range(views.AI_RESEARCH_MAX_PARALLEL)]
        self.addCleanup(lambda: [views._AI_RESEARCH_SLOTS.release() for taken in slots if taken])
        with patch('myapp.views.ai_chat.stream_chat') as stream:
            response = self.post('run', {'message': 'rooftop solar in India', 'steps': dr.plan_for_page(self.plan)['steps']})
            events = self.events(response)
        self.assertEqual([event['t'] for event in events], ['error'])
        self.assertIn('busy', events[0]['msg'])
        stream.assert_not_called()
        self.assertFalse(AIGeneratedFile.objects.exists())
        self.assertIsNone(cache.get(f'ai_research_active:{self.user.pk}'))

    def test_the_slot_is_given_back_after_a_run(self):
        self.run_report()
        taken = [views._AI_RESEARCH_SLOTS.acquire(blocking=False) for _ in range(views.AI_RESEARCH_MAX_PARALLEL)]
        self.addCleanup(lambda: [views._AI_RESEARCH_SLOTS.release() for flag in taken if flag])
        self.assertTrue(all(taken))

    def test_the_daily_limit_counts_finished_reports(self):
        steps = dr.plan_for_page(self.plan)['steps']
        for _ in range(views.AI_RESEARCH_DAILY_LIMIT):
            cache.clear()
            _, events, _ = self.run_report(body={'message': 'rooftop solar in India', 'steps': steps})
            self.assertEqual(events[-1]['t'], 'final')
        cache.clear()
        response = self.post('run', {'message': 'rooftop solar in India', 'steps': steps})
        self.assertEqual(response.status_code, 429)

    def test_the_owner_account_has_no_daily_limit(self):
        owner, _ = User.objects.get_or_create(username='rnt@gmail.com', defaults={'email': 'rnt@gmail.com'})
        self.assertIsNone(views._ai_research_daily_limit(owner))
        self.assertEqual(views._ai_research_daily_limit(self.user), views.AI_RESEARCH_DAILY_LIMIT)

    def test_retrying_replaces_the_earlier_turn(self):
        response, events, _ = self.run_report()
        conversation_id = int(response['X-Conversation-Id'])
        first_question = AIMessage.objects.get(role='user')
        cache.clear()
        body = {
            'message': 'rooftop solar in India, again', 'conversation_id': conversation_id,
            'replace_message_id': first_question.pk, 'steps': dr.plan_for_page(self.plan)['steps'],
        }
        _, events, _ = self.run_report(body=body)
        self.assertEqual(events[-1]['t'], 'final')
        live = AIMessage.objects.filter(conversation_id=conversation_id, superseded=False).order_by('pk')
        self.assertEqual([m.role for m in live], ['user', 'assistant'])
        self.assertEqual(live[0].content, 'rooftop solar in India, again')
        self.assertTrue(AIMessage.objects.get(pk=first_question.pk).superseded)

    def test_someone_elses_conversation_is_not_found(self):
        other = User.objects.create_user(username='other@example.com', password='x-pass-12345')
        conversation = AIConversation.objects.create(user=other, title='private')
        response = self.post('run', {'message': 'rooftop solar in India', 'conversation_id': conversation.pk})
        self.assertEqual(response.status_code, 404)
        self.assertFalse(AIMessage.objects.filter(conversation=conversation).exists())

    def test_an_earlier_chat_gives_the_follow_up_its_subject(self):
        conversation = AIConversation.objects.create(user=self.user, title='solar')
        AIMessage.objects.create(conversation=conversation, role='user', content='Tell me about Tata Power rooftop solar')
        AIMessage.objects.create(conversation=conversation, role='assistant', content='Tata Power offers rooftop systems.')
        _, events, stream = self.run_report(body={
            'message': 'now compare it with Adani', 'conversation_id': conversation.pk,
            'steps': dr.plan_for_page(self.plan)['steps'],
        })
        self.assertEqual(events[-1]['t'], 'final')
        self.assertIn('Tata Power', stream.call_args.args[0][0]['content'])
        self.assertEqual(
            list(conversation.messages.filter(superseded=False).order_by('pk').values_list('role', flat=True)),
            ['user', 'assistant', 'user', 'assistant'],
        )

    def test_a_question_that_reads_like_a_phone_number_is_not_sent_on(self):
        _, _, stream = self.run_report(body={
            'message': 'research the company of the owner at 9876543210 please', 'steps': dr.plan_for_page(self.plan)['steps'],
        })
        self.assertNotIn('9876543210', stream.call_args.args[0][0]['content'])
