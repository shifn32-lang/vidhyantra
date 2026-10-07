"""The deep-research screen on /AI/: the button is on the page, and the report
renderer in ai.html (the real functions, run under Node) draws a finished report
the way the server writes it. Skipped when Node is not installed."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

TEMPLATE = Path(__file__).resolve().parent / 'templates' / 'ai.html'

REPORT = """# Rooftop solar in India

> **Summary:** Rooftop solar is supported by a central subsidy [2].

## Key findings

- Households get a subsidy for small systems [2].
- Costs fell over the last decade [1][3].

## Costs

| Item | Detail |
|---|---|
| Subsidy | Central scheme [2] |

## Sources

1. [Cost study](https://one.example/a?x=1&y=2)
2. [Scheme page](https://two.example/b)
3. [Market report](https://three.example/c)

---

*Researched on 7 October 2026 from 3 web sources (3 cited).*

**Download this report:** [PDF](https://site.test/AI/api/files/aaa/download/) · [Word](https://site.test/AI/api/files/bbb/download/) · [Markdown](https://site.test/AI/api/files/ccc/download/)
"""

HARNESS = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8').split('\r\n').join('\n');
function fn(name) {
  const start = html.indexOf('\n  function ' + name + '(');
  if (start < 0) throw new Error('missing function ' + name);
  return html.slice(start, html.indexOf('\n  }\n', start + 1) + 4);
}
function variable(name) {
  const start = html.indexOf('var ' + name + ' = ');
  if (start < 0) throw new Error('missing var ' + name);
  return html.slice(start, html.indexOf('};', start) + 2);
}
const names = ['escHtml', 'cleanCodeLanguage', 'extractCodeBlocks', 'normalizeEquationText', 'isTableRowLine',
  'isTableSepLine', 'parseTableRow', 'extractTables', 'linkifyStash', 'formatMd', 'plainReplyText', 'cardLabelFor',
  'isResearchReport', 'citationLinks', 'hostOfUrl', 'downloadIconFor', 'polishReport'];
const code = variable('COPY_CARDS') + '\n' + variable('CALLOUT_KINDS') + '\n' + names.map(fn).join('\n') +
  '\nmodule.exports = { formatMd, plainReplyText, isResearchReport };';
const m = { exports: {} };
new Function('module', 'exports', code)(m, m.exports);
const { formatMd, plainReplyText, isResearchReport } = m.exports;
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = {
  isReport: isResearchReport(input.report),
  notReport: isResearchReport('# Heading\n\nJust text.'),
  report: formatMd(input.report, { report: true }),
  plain: formatMd(input.report),
  copied: plainReplyText(input.report),
  nasty: formatMd(input.nasty, { report: true }),
  unlisted: formatMd('# T\n\nClaim [3].\n\n**Download this report:** x', { report: true }),
  reply: formatMd('## Heading\n\nSome **bold** text and [a link](https://example.com).'),
};
process.stdout.write(JSON.stringify(out));
"""

NASTY = (
    '# T <img src=x onerror=alert(1)>\n\n> **Summary:** a [1]\n\n## Sources\n\n'
    '1. [Evil <script>alert(1)</script>](https://a.example/x"onmouseover="alert(1))\n\n'
    '**Download this report:** [PDF](https://site.test/AI/api/files/1/download/)\n'
)


class ResearchPageTests(TestCase):
    def test_the_page_has_the_deep_research_button_and_bar(self):
        response = self.client.get('/AI/', follow=True)
        page = response.content.decode('utf-8')
        for needle in ('id="researchBtn"', 'id="researchModeBar"', '/AI/api/research/plan/',
                       '/AI/api/research/run/', 'Deep research'):
            self.assertIn(needle, page)

    def test_a_logged_in_page_still_renders(self):
        user = User.objects.create_user(username='page@example.com', password='x-pass-12345')
        self.client.force_login(user)
        self.assertEqual(self.client.get('/AI/', follow=True).status_code, 200)


@__import__('unittest').skipUnless(shutil.which('node'), 'Node is not installed')
class ReportRendererTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder, 'input.json')
            harness = Path(folder, 'harness.js')
            data.write_text(json.dumps({'report': REPORT, 'nasty': NASTY}), encoding='utf-8')
            harness.write_text(HARNESS, encoding='utf-8')
            result = subprocess.run(
                ['node', str(harness), str(TEMPLATE), str(data)], capture_output=True, text=True, encoding='utf-8',
            )
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.out = json.loads(result.stdout)

    def test_a_report_is_recognised_and_an_ordinary_reply_is_not(self):
        self.assertTrue(self.out['isReport'])
        self.assertFalse(self.out['notReport'])

    def test_a_report_keeps_its_headings_callout_and_table(self):
        report = self.out['report']
        self.assertIn('<h1>Rooftop solar in India</h1>', report)
        self.assertIn('<h2>Key findings</h2>', report)
        self.assertIn('<blockquote class="callout-summary"><strong>Summary:</strong>', report)
        self.assertIn('msg-table-block', report)
        self.assertRegex(report, r'<h2>Sources</h2><ol>(?:<li>.*?</li>){3}</ol>')

    def test_citations_link_to_their_sources(self):
        report = self.out['report']
        self.assertEqual(report.count('<a class="cite"'), 5)
        self.assertIn('<a class="cite" href="https://two.example/b"', report)
        self.assertIn('<span class="src-host">one.example</span>', report)
        self.assertIn('https://one.example/a?x=1&amp;y=2', report)          # an address with & stays valid HTML

    def test_the_download_line_becomes_buttons_under_the_title(self):
        report = self.out['report']
        self.assertEqual(report.count('class="rd-btn"'), 3)
        self.assertGreater(report.index('research-downloads'), report.index('</h1>'))
        self.assertLess(report.index('research-downloads'), report.index('callout-summary'))
        self.assertNotIn('Download this report:</strong>', report)
        for icon in ('fa-file-pdf', 'fa-file-word', 'fa-file-lines'):
            self.assertIn(icon, report)

    def test_the_same_text_outside_report_mode_is_unchanged_chat_formatting(self):
        plain = self.out['plain']
        self.assertIn('<p><strong>Key findings</strong></p>', plain)
        self.assertNotIn('<h1>', plain)
        self.assertNotIn('class="cite"', plain)
        self.assertIn('<p><strong>Heading</strong></p>', self.out['reply'])

    def test_copying_a_report_leaves_out_the_download_line(self):
        copied = self.out['copied']
        self.assertTrue(copied.startswith('# Rooftop solar in India'))
        self.assertNotIn('Download this report', copied)
        self.assertTrue(copied.rstrip().endswith('*Researched on 7 October 2026 from 3 web sources (3 cited).*'))

    def test_markup_in_a_title_or_a_source_is_escaped(self):
        nasty = self.out['nasty']
        self.assertNotIn('<script', nasty.lower())
        self.assertNotIn('<img', nasty.lower())
        self.assertNotIn(' onmouseover=', nasty.replace('&quot;', ''))

    def test_a_citation_without_a_listed_source_stays_plain(self):
        self.assertIn('Claim [3].', self.out['unlisted'])
