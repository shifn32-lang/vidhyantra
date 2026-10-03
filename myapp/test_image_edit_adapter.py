import base64
import io
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings
from PIL import Image

from myapp import ai_chat, image_generation
from myapp.views import _ai_public_routed_model_key, _chatgpt_image_error_detail


class ImageEditAdapterTests(SimpleTestCase):
    @override_settings(NVIDIA_FLUX_API_KEY='primary', NVIDIA_FLUX_BACKUP_API_KEY='secondary',
                       NVIDIA_FLUX_DEV_API_KEY='dev')
    @patch('myapp.image_generation.requests.post')
    def test_a_failed_generation_is_reported_without_any_fallback(self, post):
        post.return_value = Mock(status_code=503, **{'json.return_value': {}})
        with self.assertRaises(image_generation.ImageGenerationError):
            image_generation.generate_image('A blue mug')
        # One request, on the one configured key: no backup key, no FLUX.1-dev.
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer primary')

    @patch('myapp.image_generation._dispatch_generate')
    def test_a_blocked_prompt_is_not_retried_on_another_provider(self, dispatch):
        dispatch.side_effect = image_generation.ImageGenerationError('Blocked', status_code=400, blocked=True)
        with self.assertRaises(image_generation.ImageGenerationError):
            image_generation.generate_image('A blue mug', model_key='flux-klein-4b')
        dispatch.assert_called_once()

    @override_settings(QWEN_IMAGE_EDIT_API_URL='')
    @patch('myapp.image_generation.requests.post')
    def test_qwen_requires_deployment(self, post):
        with self.assertRaises(image_generation.ImageGenerationError) as raised:
            image_generation.generate_image('Make it blue', 'image', model_key='qwen-image-edit')
        self.assertIn('not connected', str(raised.exception))
        post.assert_not_called()

    @override_settings(QWEN_IMAGE_EDIT_API_URL='http://localhost:8002/v1/infer', QWEN_IMAGE_EDIT_ENDPOINT_KEY='')
    @patch('myapp.image_generation._decode_artifact')
    @patch('myapp.image_generation.requests.post')
    def test_qwen_sends_uploaded_image_to_nim(self, post, decode):
        source = io.BytesIO()
        Image.new('RGB', (64, 64), 'red').save(source, 'PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()
        post.return_value.status_code = 200
        image_generation.generate_image('Make it blue', uri, model_key='qwen-image-edit')
        self.assertEqual(post.call_args.args[0], 'http://localhost:8002/v1/infer')
        self.assertNotIn('Authorization', post.call_args.kwargs['headers'])
        self.assertTrue(post.call_args.kwargs['json']['image'].startswith('data:image/jpeg;base64,'))
        decode.assert_called_once()

    @override_settings(NVIDIA_FLUX_KONTEXT_API_KEY='kontext-test', FLUX_EDIT_API_URL='')
    @patch('myapp.image_generation._decode_artifact')
    @patch('myapp.image_generation.requests.post')
    def test_kontext_uses_own_key_and_single_image_schema(self, post, decode):
        source = io.BytesIO()
        Image.new('RGB', (64, 64), 'red').save(source, 'PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()
        post.return_value.status_code = 200
        image_generation.generate_image('Make it blue', uri, model_key='flux-kontext-dev')
        request = post.call_args
        self.assertTrue(request.args[0].endswith('flux.1-kontext-dev'))
        self.assertEqual(request.kwargs['headers']['Authorization'], 'Bearer kontext-test')
        body = request.kwargs['json']
        self.assertIsInstance(body['image'], str)
        self.assertEqual(body['steps'], 30)
        self.assertNotIn('width', body)
        self.assertNotIn('height', body)

    def test_kontext_requires_source_image(self):
        with self.assertRaises(image_generation.ImageGenerationError) as raised:
            image_generation.generate_image('Draw a cat', model_key='flux-kontext-dev')
        self.assertEqual(raised.exception.status_code, 400)

    @override_settings(FLUX_EDIT_API_URL='http://localhost:8001/v1/infer', FLUX_EDIT_API_KEY='')
    @patch('myapp.image_generation._decode_artifact')
    @patch('myapp.image_generation.requests.post')
    def test_upload_is_converted_and_sent_to_edit_server_without_hosted_key(self, post, decode):
        source = io.BytesIO()
        Image.new('RGBA', (2048, 1024), (255, 0, 0, 128)).save(source, 'PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()
        post.return_value.status_code = 200
        image_generation.generate_image('change name to sindore', uri)
        self.assertEqual(post.call_args.args[0], 'http://localhost:8001/v1/infer')
        self.assertNotIn('Authorization', post.call_args.kwargs['headers'])
        encoded = post.call_args.kwargs['json']['image'][0].split(',', 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as converted:
            self.assertEqual(converted.mode, 'RGB')
            self.assertEqual(converted.size, (1024, 512))
        decode.assert_called_once()

    @override_settings(FLUX_EDIT_API_URL='', NVIDIA_FLUX_EDIT_API_KEY='hosted-preview-key')
    @patch('myapp.image_generation.requests.post')
    def test_hosted_preview_is_not_sent_arbitrary_uploads(self, post):
        source = io.BytesIO()
        Image.new('RGB', (64, 64), 'red').save(source, 'PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(source.getvalue()).decode()

        with self.assertRaises(image_generation.ImageGenerationError) as raised:
            image_generation.generate_image('Make it blue', uri)

        self.assertTrue(raised.exception.editing_unavailable)
        post.assert_not_called()

    def test_preview_restriction_does_not_blame_file_size(self):
        response = Mock(status_code=422)
        response.json.return_value = {'detail': 'expected: example_id'}
        error = image_generation._safe_error(response)
        self.assertEqual(error.status_code, 503)
        self.assertNotIn('smaller', str(error))

    def test_gpt_names_survive_image_routing(self):
        for key in (ai_chat.CHATGPT_56_MODEL_KEY, 'gpt-oss-20b'):
            self.assertEqual(_ai_public_routed_model_key(key, ai_chat.FLUX_KLEIN_4B_MODEL_KEY), key)
            error = image_generation.ImageGenerationError('NVIDIA unavailable')
            self.assertIn(ai_chat.MODELS[key]['label'], _chatgpt_image_error_detail(error, key))
        self.assertTrue(ai_chat.is_image_edit_instruction('change name to sindore'))
