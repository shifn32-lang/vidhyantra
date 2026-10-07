"""Image requests: a picture request is never turned into a text file, and a
prompt the image service's filter refuses (because of a famous brand name) is
retried once with the brand described in plain words."""
from unittest.mock import patch

from django.test import SimpleTestCase

from myapp import image_generation as ig, views

BRAND_POSTER = (
    "Create a image for brand name Edu AI – Advanced AI Solutions, featuring AI Models, Image Generation, "
    "File Upload, Web Search & GitHub connectivity, with plans starting at ₹99. Include Email: "
    "support@edutrellis.in, Contact: 9695953183, and Website: edutrellis.in. Use a futuristic dark "
    "AI-themed background. Create in 1080×1350 Instagram portrait size"
)


class PictureIsNotAFileTests(SimpleTestCase):
    def test_picture_requests_that_mention_files_or_addresses_stay_pictures(self):
        for text in (
            BRAND_POSTER,
            'create a image of iphone 16',
            'create an image of a document on a desk',
            'make a poster with the words File Upload on it',
        ):
            with self.subTest(text=text[:40]):
                self.assertIsNone(views._ai_generated_file_spec(text))

    def test_real_file_requests_still_make_files(self):
        self.assertEqual(views._ai_generated_file_spec('make a file with my notes')['file_name'], 'generated.txt')
        self.assertEqual(views._ai_generated_file_spec('create a pdf with an image of a logo')['file_name'], 'generated.pdf')
        self.assertEqual(views._ai_generated_file_spec('generate an image and save it as poster.pdf')['file_name'], 'poster.pdf')
        self.assertEqual(views._ai_generated_file_spec('create an excel sheet of prices')['file_name'], 'generated.xlsx')


class BrandRetryTests(SimpleTestCase):
    def blocked(self):
        return ig.ImageGenerationError('blocked', status_code=400, blocked=True)

    def test_built_in_swaps_describe_famous_brands_in_plain_words(self):
        with patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')):
            self.assertEqual(
                ig.brand_neutral_prompt('create a image of iphone 16'), 'create a image of modern flagship smartphone',
            )
            self.assertEqual(ig.brand_neutral_prompt('nike running shoes'), 'running shoes')
            self.assertEqual(ig.brand_neutral_prompt('a red tesla model 3'), 'a red electric car')
            self.assertEqual(ig.brand_neutral_prompt('a calm forest'), 'a calm forest')

    def test_the_model_rewrite_is_used_when_it_answers(self):
        with patch('myapp.ai_chat.complete_json', return_value={'prompt': 'a sleek smartphone, studio photo'}):
            self.assertEqual(ig.brand_neutral_prompt('iphone 16 studio photo'), 'a sleek smartphone, studio photo')

    def test_a_filtered_prompt_is_retried_once_in_plain_words(self):
        calls = []

        def dispatch(prompt, source):
            calls.append(prompt)
            if 'iphone' in prompt.lower():
                raise self.blocked()
            return ig.GeneratedImage(content=b'x', extension='jpg')

        with patch('myapp.image_generation._dispatch_generate', side_effect=dispatch), \
                patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')):
            image = ig.generate_image('create a image of iphone 16')
        self.assertEqual(image.extension, 'jpg')
        self.assertEqual(len(calls), 2)
        self.assertNotIn('iphone', calls[1].lower())

    def test_when_the_retry_is_refused_too_the_original_error_is_reported(self):
        with patch('myapp.image_generation._dispatch_generate', side_effect=self.blocked()) as dispatch, \
                patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')):
            with self.assertRaises(ig.ImageGenerationError) as caught:
                ig.generate_image('create a image of iphone 16')
        self.assertTrue(caught.exception.blocked)
        self.assertEqual(dispatch.call_count, 2)

    def test_nothing_changes_for_other_failures_or_nothing_to_swap(self):
        with patch('myapp.image_generation._dispatch_generate',
                   side_effect=ig.ImageGenerationError('down')) as dispatch:
            with self.assertRaises(ig.ImageGenerationError):
                ig.generate_image('a calm forest')
        self.assertEqual(dispatch.call_count, 1)
        with patch('myapp.image_generation._dispatch_generate', side_effect=self.blocked()) as dispatch, \
                patch('myapp.ai_chat.complete_json', side_effect=RuntimeError('down')):
            with self.assertRaises(ig.ImageGenerationError):
                ig.generate_image('a calm forest')
        self.assertEqual(dispatch.call_count, 1)


class FollowUpOnAShownPictureTests(SimpleTestCase):
    def test_a_few_words_about_the_look_are_an_edit(self):
        from myapp import ai_chat
        for text in ('hands up pose', 'red dress', 'make her smile', 'beach background', 'standing near a car',
                     'more realistic', 'anime style'):
            with self.subTest(text=text):
                self.assertTrue(ai_chat.is_image_followup(text))

    def test_questions_and_chit_chat_are_not(self):
        from myapp import ai_chat
        for text in ('why is the background blurry?', 'thanks', 'nice', 'what pose is this', 'hi',
                     'write a caption for this photo', 'tell me about the dress',
                     'this is a long message about how I spent my day at the beach with my friends and family'):
            with self.subTest(text=text):
                self.assertFalse(ai_chat.is_image_followup(text))
