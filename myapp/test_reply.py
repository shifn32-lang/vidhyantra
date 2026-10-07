"""The Reply button under an AI answer: the quoted answer is saved with the
message, shown again when the chat is reopened, and handed to the model."""
import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase

from myapp import views
from myapp.models import AIConversation, AIMessage


class ReplyQuoteTests(SimpleTestCase):
    def test_a_quote_is_tidied_and_capped(self):
        quote = views._ai_reply_quote({'reply_to': '  Paris   is\tthe capital.\r\n\r\n\r\n\r\nIt is big.  '})
        self.assertEqual(quote, 'Paris is the capital.\n\nIt is big.')
        long_quote = views._ai_reply_quote({'reply_to': 'word ' * 1000})
        self.assertLessEqual(len(long_quote), views.AI_REPLY_QUOTE_MAX_CHARS)

    def test_anything_that_is_not_text_is_not_a_reply(self):
        for value in (None, 12, ['a'], {'a': 1}, ''):
            self.assertEqual(views._ai_reply_quote({'reply_to': value}), '')
        self.assertEqual(views._ai_reply_quote({}), '')


class ReplyChatTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            username='reply-tests@example.com', password='test-password-123', is_staff=True,
        )
        self.client.force_login(self.user)
        self.conversation = AIConversation.objects.create(user=self.user, title='Capitals')
        AIMessage.objects.create(conversation=self.conversation, role='user', content='List capitals')
        AIMessage.objects.create(
            conversation=self.conversation, role='assistant', model_key='quick',
            content='Kosovo, Taiwan and Western Sahara are not included in this list.',
        )

    def send(self, **extra):
        seen = {}

        def fake_stream(history, *args, **kwargs):
            seen['history'] = history
            return iter(['Sure.'])

        body = {'conversation_id': self.conversation.pk, 'message': 'why not?', 'model': 'quick'}
        body.update(extra)
        with patch('myapp.views.ai_chat.stream_chat', side_effect=fake_stream):
            response = self.client.post('/AI/api/send/', data=json.dumps(body), content_type='application/json')
            response.getvalue()   # drains the streamed reply so the turn is fully saved
        return response, seen.get('history') or []

    def test_the_model_is_told_which_answer_the_message_is_about(self):
        response, history = self.send(reply_to='Kosovo, Taiwan and Western Sahara are not included')
        self.assertEqual(response.status_code, 200)
        last = history[-1]
        self.assertEqual(last['role'], 'user')
        self.assertIn('replying to this specific earlier answer', last['content'])
        self.assertIn('Kosovo, Taiwan and Western Sahara are not included', last['content'])
        self.assertTrue(last['content'].rstrip().endswith('why not?'))

    def test_the_quote_is_saved_and_comes_back_when_the_chat_is_reopened(self):
        self.send(reply_to='Kosovo is not included')
        sent = AIMessage.objects.filter(role='user').latest('pk')
        self.assertEqual(sent.content, 'why not?')
        self.assertEqual(sent.reply_to_text, 'Kosovo is not included')
        messages = self.client.get(f'/AI/api/conversations/{self.conversation.pk}/').json()['messages']
        self.assertEqual(messages[-2]['reply_to_text'], 'Kosovo is not included')
        self.assertEqual(messages[0]['reply_to_text'], '')

    def test_later_turns_keep_the_quote_in_the_history(self):
        self.send(reply_to='Kosovo is not included')
        _, history = self.send(message='and Taiwan?')
        quoted = [m for m in history if m['role'] == 'user' and 'Kosovo is not included' in m['content']]
        self.assertEqual(len(quoted), 1)
        self.assertEqual(history[-1]['content'], 'and Taiwan?')

    def test_an_ordinary_message_is_sent_exactly_as_typed(self):
        _, history = self.send()
        self.assertEqual(history[-1]['content'], 'why not?')
        self.assertEqual(AIMessage.objects.filter(role='user').latest('pk').reply_to_text, '')


class FollowUpOnAShownPictureChatTests(TestCase):
    PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='followup@example.com', password='test-password-123', is_staff=True)
        self.client.force_login(self.user)
        self.conversation = AIConversation.objects.create(user=self.user, title='Photo')
        AIMessage.objects.create(conversation=self.conversation, role='user', content='girl sitting on a bench')

    def send(self, message):
        from django.http import HttpResponse
        with patch('myapp.views._ai_flux_response', return_value=HttpResponse()) as flux, \
                patch('myapp.views.ai_chat.stream_chat', return_value=iter(['ok'])):
            self.client.post('/AI/api/send/', data=json.dumps({
                'conversation_id': self.conversation.pk, 'message': message, 'model': 'quick',
            }), content_type='application/json').getvalue()
        return flux

    def test_right_after_a_picture_a_look_instruction_edits_it(self):
        AIMessage.objects.create(conversation=self.conversation, role='assistant', content='', image_data=self.PNG, model_key='quick')
        flux = self.send('hands up pose')
        flux.assert_called_once()
        self.assertEqual(flux.call_args.args[2], self.PNG)

    def test_once_the_chat_has_moved_on_it_is_ordinary_chat(self):
        AIMessage.objects.create(conversation=self.conversation, role='assistant', content='', image_data=self.PNG, model_key='quick')
        AIMessage.objects.create(conversation=self.conversation, role='user', content='tell me a joke')
        AIMessage.objects.create(conversation=self.conversation, role='assistant', content='A joke.', model_key='quick')
        self.send('red dress').assert_not_called()
