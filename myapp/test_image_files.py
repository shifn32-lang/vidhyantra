"""Three chat fixes: an image from the chat put into a PDF/Word/PowerPoint
file, posters asked for without "create"/"image", and generated files that
carry no brand name."""
import base64
import io
import json
import zipfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from PIL import Image

from myapp import ai_chat, doc_image, views
from myapp.models import AIConversation, AIGeneratedFile, AIMessage


def png_data_uri(size=(80, 100), colour=(30, 60, 200)):
    out = io.BytesIO()
    Image.new('RGB', size, colour).save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()


class ImageFileRequestTests(SimpleTestCase):
    def test_putting_a_chat_image_into_a_file_is_recognised(self):
        for text, ext in (
            ('add this image inside pdf and generate pdf downlaod link', 'pdf'),
            ('put the poster in a word file', 'docx'),
            ('make a ppt of this image', 'pptx'),
            ('convert it into pdf', 'pdf'),
            ('save the above image as poster.pdf', 'pdf'),
        ):
            with self.subTest(text=text):
                self.assertEqual(views._ai_image_file_request(text), ext)

    def test_written_documents_are_not(self):
        for text in (
            'create a pdf with an image of a logo', 'make a pdf about the image generation industry',
            'add this image', 'create a pdf report on India', 'what is a pdf',
        ):
            with self.subTest(text=text):
                self.assertEqual(views._ai_image_file_request(text), '')


class ImageFileBuildTests(SimpleTestCase):
    def test_each_format_holds_the_picture(self):
        content = doc_image.stored_content(png_data_uri())
        pdf = doc_image.render('poster.pdf', content)
        self.assertTrue(pdf.startswith(b'%PDF'))
        import fitz
        page = fitz.open(stream=pdf, filetype='pdf')[0]
        self.assertEqual(len(page.get_images()), 1)
        self.assertAlmostEqual(page.rect.height / page.rect.width, 100 / 80, places=2)
        for name in ('poster.docx', 'poster.pptx'):
            with zipfile.ZipFile(io.BytesIO(doc_image.render(name, content))) as package:
                self.assertTrue(any('/media/' in n for n in package.namelist()), name)


class ImageFileChatTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='imgfile@example.com', password='test-password-123', is_staff=True)
        self.client.force_login(self.user)
        self.conversation = AIConversation.objects.create(user=self.user, title='Poster')
        AIMessage.objects.create(conversation=self.conversation, role='user', content='poster for my cafe')

    def send(self, message, **extra):
        body = {'conversation_id': self.conversation.pk, 'message': message, 'model': 'sol'}
        body.update(extra)
        with patch('myapp.views.ai_chat.stream_chat', side_effect=AssertionError('no model call')), \
                patch('myapp.views._ai_flux_response', side_effect=AssertionError('no image call')):
            return self.client.post('/AI/api/send/', data=json.dumps(body), content_type='application/json')

    def test_the_last_image_goes_into_a_pdf_with_a_download_link(self):
        AIMessage.objects.create(conversation=self.conversation, role='assistant', content='', image_data=png_data_uri(), model_key='sol')
        response = self.send('add this image inside pdf and generate pdf downlaod link')
        self.assertEqual(response.status_code, 200)
        reply = response.content.decode()
        self.assertIn('[Download image.pdf]', reply)
        saved = AIGeneratedFile.objects.get()
        self.assertIn(str(saved.token), reply)
        download = self.client.get(reply.split('](')[1].rstrip(')').replace('http://testserver', ''))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download['Content-Type'], 'application/pdf')
        self.assertTrue(download.content.startswith(b'%PDF'))
        self.assertEqual(AIMessage.objects.filter(role='assistant').last().content, reply)

    def test_a_poster_into_word_is_named_after_it(self):
        AIMessage.objects.create(conversation=self.conversation, role='assistant', content='', image_data=png_data_uri(), model_key='sol')
        reply = self.send('put this poster in a word file').content.decode()
        self.assertIn('[Download poster.docx]', reply)

    def test_with_no_image_in_the_chat_it_says_so(self):
        reply = self.send('add this image inside pdf').content.decode()
        self.assertIn("There's no image in this chat yet", reply)
        self.assertFalse(AIGeneratedFile.objects.exists())


class DesignRequestTests(SimpleTestCase):
    def test_designs_named_without_create_or_image_are_pictures(self):
        for text in (
            'Instagram poster 4:5 aspect ratio (1080×1350), "NOVA AI – Advanced AI Platform" bold headline, '
            'dark background --ar 4:5 --stylize 750 --v 6.1',
            'Diwali sale poster with 50% off', 'need a flyer for my yoga class',
            'Instagram post for my bakery, cakes from ₹299', 'youtube thumbnail for my cooking video',
            'a cat in space --ar 16:9',
        ):
            with self.subTest(text=text[:40]):
                self.assertTrue(ai_chat.is_design_request(text))

    def test_talk_about_designs_stays_chat(self):
        for text in (
            'what is a poster?', 'how to make a poster in canva', 'write a prompt for a poster',
            'explain the poster design principles', 'I saw a poster yesterday and liked it',
            'what should I write in my instagram post', 'my story is about a dragon', 'can you check my logo',
        ):
            with self.subTest(text=text):
                self.assertFalse(ai_chat.is_design_request(text))
                self.assertFalse(ai_chat.is_image_generation_request(text))

    def test_a_named_poster_is_not_turned_into_a_text_file(self):
        self.assertIsNone(views._ai_generated_file_spec('Instagram poster with File Upload and Web Search features, make it bold'))


class NoBrandInGeneratedFilesTests(SimpleTestCase):
    TEXT = '# Quarterly Plan\n\nSome text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n'

    def test_writers_leave_out_the_brand_by_default(self):
        from myapp import doc_pdf, doc_slides, doc_text, doc_word
        import fitz
        pdf = fitz.open(stream=doc_pdf.render_pdf(self.TEXT), filetype='pdf')
        self.assertNotIn('Vidhyora', ''.join(page.get_text() for page in pdf))
        self.assertNotIn('Vidhyora', json.dumps(pdf.metadata))
        self.assertIn('Quarterly Plan', pdf[0].get_text())
        for data in (doc_word.render_docx(self.TEXT), doc_slides.render_pptx(self.TEXT)):
            with zipfile.ZipFile(io.BytesIO(data)) as package:
                text = ''.join(package.read(n).decode('utf-8', 'ignore') for n in package.namelist() if n.endswith('.xml'))
            self.assertNotIn('Vidhyora', text)
            self.assertNotIn('VIDHYORA', text)
        self.assertNotIn('Vidhyora', doc_text.render_txt(self.TEXT))


class MoreImagesGalleryTests(TestCase):
    def test_every_style_has_a_preview_and_is_drawn_by_the_image_model(self):
        import os
        from django.conf import settings
        from myapp import image_styles
        for style in image_styles.gallery():
            with self.subTest(style=style['id']):
                self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'myapp', 'static', style['thumb'])))
                if style['photo']:
                    self.assertTrue(ai_chat.is_image_edit_instruction(style['prompt']))
                else:
                    text = style['prompt'] + 'a red bicycle'
                    self.assertTrue(ai_chat.is_image_generation_request(text))
                    self.assertIsNone(views._ai_generated_file_spec(text))

    def test_the_page_shows_the_gallery(self):
        page = self.client.get('/AI/', secure=True, follow=True).content.decode()
        self.assertIn('id="moreImagesBtn"', page)
        self.assertIn('class="style-card"', page)
        welcome = page.split('id="emptyState"')[1].split('class="start-grid"')[0]
        self.assertNotIn('brand-mark', welcome)     # the logo is only in the header now
        self.assertIn('empty-greet', welcome)
