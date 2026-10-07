"""Posters with exact words: the image model paints only the artwork and the
server letters the user's own words on top (myapp/poster.py)."""
import io
from unittest.mock import patch

from django.test import SimpleTestCase
from PIL import Image

from myapp import image_generation as ig, poster

EDU_POSTER = (
    "Create a image for brand name Edu AI – Advanced AI Solutions, featuring AI Models, Image Generation, "
    "File Upload, Web Search & GitHub connectivity, with plans starting at ₹99. Include Email: "
    "support@edutrellis.in, Contact: 9695953183, and Website: edutrellis.in. Use a futuristic dark AI-themed "
    "background with glowing technology elements, clean premium typography, professional branding, strong "
    "visual hierarchy, and highly readable text. Create in 1080×1350 Instagram portrait size"
)


def artwork(size=(896, 1120)):
    image = Image.new('RGB', size, (20, 30, 70))
    for x in range(0, size[0], 40):
        for y in range(0, size[1], 40):
            image.putpixel((x, y), (60, 160, 255))
    out = io.BytesIO()
    image.save(out, format='PNG')
    return ig.GeneratedImage(content=out.getvalue(), extension='png')


class WhichRequestsArePostersTests(SimpleTestCase):
    def test_pictures_with_exact_details_are_posters(self):
        for text in (
            EDU_POSTER,
            'Make an instagram post for my gym FitZone: membership ₹999/month, call 9876543210',
            'design a flyer for my shop, email hello@shop.in',
        ):
            with self.subTest(text=text[:40]):
                self.assertTrue(poster.wants_poster(text))

    def test_ordinary_pictures_are_not(self):
        for text in (
            'create a image of iphone 16', 'a cat sitting on the moon', 'draw a sunset over mountains',
            'make a logo for my brand', 'what is the price of ₹99 plan?',
        ):
            with self.subTest(text=text):
                self.assertFalse(poster.wants_poster(text))


class ReadingTheWordsTests(SimpleTestCase):
    def test_contact_details_are_copied_exactly_by_pattern(self):
        emails, phones, websites = poster._contact_details(EDU_POSTER)
        self.assertEqual(emails, ['support@edutrellis.in'])
        self.assertEqual(phones, ['9695953183'])
        self.assertEqual(websites, ['edutrellis.in'])
        self.assertEqual(poster._contact_details('call +91 98765 43210 now')[1], ['+91 98765 43210'])

    def test_without_the_model_the_words_still_come_from_the_prompt(self):
        with patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')):
            spec = poster.extract_spec(EDU_POSTER)
        self.assertEqual(spec.title, 'Edu AI')
        self.assertEqual(spec.tagline, 'Advanced AI Solutions')
        self.assertEqual(spec.features, ['AI Models', 'Image Generation', 'File Upload', 'Web Search & GitHub connectivity'])
        self.assertEqual(spec.offer, 'Plans starting at ₹99')
        self.assertNotRegex(spec.scene.lower(), r'typography|readable|text|1080')

    def test_words_the_model_invents_are_dropped(self):
        reply = {
            'title': 'Edu AI', 'tagline': 'The Smartest Learning Platform Ever',
            'features': ['AI Models', 'Free Lifetime Updates'], 'offer': 'Plans starting at ₹49',
            'cta': '', 'scene': 'dark glowing circuits',
        }
        with patch('myapp.ai_chat.complete_json', return_value=reply):
            spec = poster.extract_spec(EDU_POSTER)
        self.assertEqual(spec.tagline, 'Advanced AI Solutions')   # the invented one is replaced by the user's own
        self.assertNotIn('Free Lifetime Updates', spec.features)
        self.assertEqual(spec.offer, 'Plans starting at ₹99')       # a wrong price never gets printed

    def test_a_heading_with_a_dash_becomes_title_and_tagline_and_contacts_are_not_repeated(self):
        reply = {
            'title': 'Sharma Bakery – Fresh Cakes Daily', 'tagline': '', 'features': [],
            'offer': 'cakes from ₹299', 'cta': 'Call +91 98765 43210, visit sharmabakery.com', 'scene': 'cakes',
        }
        prompt = 'Design a banner for Sharma Bakery – Fresh Cakes Daily, with cakes from ₹299. Call +91 98765 43210, visit sharmabakery.com'
        with patch('myapp.ai_chat.complete_json', return_value=reply):
            spec = poster.extract_spec(prompt)
        self.assertEqual((spec.title, spec.tagline), ('Sharma Bakery', 'Fresh Cakes Daily'))
        self.assertEqual(spec.offer, 'Cakes from ₹299')
        self.assertEqual(spec.cta, '')

    def test_the_background_prompt_asks_for_no_writing(self):
        prompt = poster.background_prompt(poster.PosterSpec(scene='neon city'))
        self.assertIn('No text', prompt)
        self.assertNotIn('Edu AI', prompt)


class MakingThePosterTests(SimpleTestCase):
    def test_the_poster_is_the_exact_size_asked_for(self):
        sent = {}

        def fake_generate(prompt, source_image=None, *, size=None):
            sent.update(prompt=prompt, size=size)
            return artwork()

        with patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')), \
                patch('myapp.poster.image_generation.generate_image', side_effect=fake_generate):
            result = poster.make_poster(EDU_POSTER)
        self.assertEqual(result.extension, 'jpg')
        self.assertEqual(Image.open(io.BytesIO(result.content)).size, (1080, 1350))
        self.assertEqual(sent['size'], (896, 1120))
        for word in ('Edu AI', 'support@edutrellis.in', '9695953183', '₹99'):
            self.assertNotIn(word, sent['prompt'])      # the image model never sees the words

    def test_layouts_fit_on_wide_square_and_crowded_posters(self):
        spec = poster.PosterSpec(
            title='A Rather Long Business Name For Testing', tagline='With a long tagline that needs wrapping too',
            features=['First feature', 'Second feature here', 'Third', 'Fourth feature', 'Fifth one', 'Sixth'],
            offer='Plans starting at ₹99', emails=['someone.long@example-company.in'], phones=['+91 98765 43210'],
            websites=['www.example-company.in'],
        )
        background = Image.open(io.BytesIO(artwork().content))
        for size in ((1080, 1350), (1920, 1080), (1080, 1080), (1080, 1920)):
            with self.subTest(size=size):
                out = Image.open(io.BytesIO(poster.compose(background, spec, size)))
                self.assertEqual(out.size, size)

    def test_ordinary_and_hindi_requests_go_the_usual_way(self):
        with patch('myapp.poster.image_generation.generate_image') as generate:
            self.assertIsNone(poster.make_poster('a cat on the moon'))
            with patch('myapp.poster.features.check', return_value=False):
                self.assertIsNone(poster.make_poster('दीपावली सेल poster, offer ₹499, call 9876543210'))
        generate.assert_not_called()

    def test_emoji_the_font_cannot_draw_are_left_out(self):
        self.assertEqual(poster.printable('Big Sale 🎉 today'), 'Big Sale today')
        self.assertEqual(poster.printable('Plans from ₹99'), 'Plans from ₹99')

    def test_image_service_failures_are_reported_as_before(self):
        with patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')), \
                patch('myapp.poster.image_generation.generate_image',
                      side_effect=ig.ImageGenerationError('busy', status_code=429)):
            with self.assertRaises(ig.ImageGenerationError):
                poster.make_poster(EDU_POSTER)


class SizeOptionTests(SimpleTestCase):
    def test_a_given_size_overrides_the_prompt(self):
        captured = {}

        def fake_dispatch(prompt, source_image, size=None):
            captured['size'] = size
            return artwork()

        with patch('myapp.image_generation._dispatch_generate', side_effect=fake_dispatch):
            ig.generate_image('neon city', size=(896, 1120))
        self.assertEqual(captured['size'], (896, 1120))
