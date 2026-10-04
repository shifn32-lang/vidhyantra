from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from django.test import SimpleTestCase
from openai import APITimeoutError

from myapp import ai_chat


class ChatLatencyTests(SimpleTestCase):
    @patch.dict(ai_chat.MODELS['quick'], {'timeout': 15.0, 'retry_attempts': 1})
    @patch('myapp.ai_chat._get_client')
    def test_a_model_timeout_is_retried_once_on_the_same_model(self, get_client):
        stalled, retry = Mock(), Mock()
        stalled.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request('POST', 'https://example.test/chat'))
        retry.chat.completions.create.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='4'))])
        ])
        get_client.side_effect = [stalled, retry]
        answer = ''.join(ai_chat.stream_chat(
            [{'role': 'user', 'content': 'What is 2+2?'}], model_key='quick'))
        self.assertEqual(answer, '4')
        model_id = ai_chat.MODELS['quick']['id']
        self.assertEqual(stalled.chat.completions.create.call_args.kwargs['model'], model_id)
        self.assertEqual(retry.chat.completions.create.call_args.kwargs['model'], model_id)
        self.assertEqual(stalled.chat.completions.create.call_args.kwargs['timeout'], 15)
