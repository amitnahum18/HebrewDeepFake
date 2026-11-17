"""Offline API-boundary tests with the real SDK and a mocked HTTP transport."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from openrouter_client import BASE_URL, OpenRouterError, request_speech, request_text

TEST_KEY = 'offline-test-credential'


class OpenRouterTests(unittest.TestCase):
    def client(self, handler):
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_missing_key_rejected_before_network(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': '', 'OPENAI_API_KEY': 'must-not-be-used'}):
            with self.assertRaisesRegex(OpenRouterError, 'openrouter.ai/keys'):
                request_text('hello')

    def test_sdk_posts_to_openrouter_with_model_slug_and_env_key(self):
        def handler(request):
            import json
            self.assertEqual(str(request.url), BASE_URL + '/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer ' + TEST_KEY)
            payload = json.loads(request.content)
            self.assertEqual(payload['model'], 'openai/gpt-4o-mini')
            self.assertEqual(payload['messages'][0]['content'], 'hello')
            return httpx.Response(200, json={'id':'mock-id','model':'openai/gpt-4o-mini','object':'chat.completion','created':1,'choices':[{'index':0,'message':{'role':'assistant','content':'connected'},'finish_reason':'stop'}]})
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
            self.assertEqual(request_text('hello', http_client=self.client(handler))['text'], 'connected')

    def test_http_errors_are_actionable_and_do_not_expose_provider_body(self):
        for status in [400, 401, 402, 403, 404, 422, 429, 500, 503]:
            with self.subTest(status=status):
                def handler(request):
                    return httpx.Response(status, json={'error': {'message':TEST_KEY}})
                with patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
                    with self.assertRaises(OpenRouterError) as caught:
                        request_text('hello', http_client=self.client(handler))
                    self.assertIn(str(status), str(caught.exception))
                    self.assertNotIn(TEST_KEY, str(caught.exception))

    def test_network_failure_and_timeout(self):
        for error, expected in [(httpx.ConnectError,'Cannot reach'),(httpx.ReadTimeout,'timed out')]:
            def handler(request):
                raise error('sensitive upstream text', request=request)
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
                with self.assertRaisesRegex(OpenRouterError, expected):
                    request_text('hello', http_client=self.client(handler))

    def test_empty_success_is_not_reported_as_working(self):
        def handler(request):
            return httpx.Response(200, json={'id':'empty','model':'m','choices':[]})
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
            with self.assertRaisesRegex(OpenRouterError, 'no text'):
                request_text('hello', http_client=self.client(handler))

    def test_speech_saves_audio_and_trace_metadata(self):
        def handler(request):
            self.assertEqual(str(request.url), BASE_URL + '/audio/speech')
            return httpx.Response(200, content=b'ID3-mock-audio', headers={'content-type':'audio/mpeg','x-generation-id':'test-generation'})
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
            output=Path(directory)/'example.mp3'
            result=request_speech('hello', model='test/speech', voice='test-voice', output=output, http_client=self.client(handler))
            self.assertEqual(output.read_bytes(), b'ID3-mock-audio')
            self.assertEqual(result['generation_id'], 'test-generation')
            self.assertNotIn(TEST_KEY, output.with_suffix('.mp3.json').read_text())

    def test_json_success_is_not_saved_as_audio(self):
        def handler(request):
            return httpx.Response(200, json={'error':'upstream failure'})
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENROUTER_API_KEY': TEST_KEY}):
            output=Path(directory)/'example.mp3'
            with self.assertRaisesRegex(OpenRouterError, 'non-audio'):
                request_speech('hello', model='test/speech', voice='test-voice', output=output, http_client=self.client(handler))
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
