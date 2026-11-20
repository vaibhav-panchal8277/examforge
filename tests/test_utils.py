from types import SimpleNamespace

import pytest
from openai import BadRequestError

from utils import create_json_completion, extract_json


@pytest.mark.parametrize("text", [
    '{"topics": [{"topic_name": "A"}]}',
    '```json\n{"topics": [{"topic_name": "A"}]}\n```',
    'Sure! Here you go:\n{"topics": [{"topic_name": "A"}]}\nHope that helps.',
])
def test_extract_json(text):
    assert extract_json(text) == {"topics": [{"topic_name": "A"}]}


def _bad_request() -> BadRequestError:
    # Build without __init__: its response type differs across openai versions.
    err = BadRequestError.__new__(BadRequestError)
    Exception.__init__(err, "response_format not supported")
    return err


class FakeClient:
    def __init__(self, first_error=None):
        self.calls = []
        self.first_error = first_error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.first_error is not None and len(self.calls) == 1:
            raise self.first_error
        return "ok"


def test_uses_json_mode_first():
    client = FakeClient()
    assert create_json_completion(client, model="m") == "ok"
    assert client.calls == [{"model": "m", "response_format": {"type": "json_object"}}]


def test_retries_without_json_mode_on_bad_request():
    client = FakeClient(first_error=_bad_request())
    assert create_json_completion(client, model="m") == "ok"
    assert len(client.calls) == 2
    assert "response_format" not in client.calls[1]


def test_other_errors_are_not_retried():
    client = FakeClient(first_error=RuntimeError("invalid API key"))
    with pytest.raises(RuntimeError, match="invalid API key"):
        create_json_completion(client, model="m")
    assert len(client.calls) == 1


from utils import get_llm_client


def test_clients_have_timeout_and_limited_retries():
    import settings
    for args in [(True, "sk-test"), (False, "")]:
        client, _ = get_llm_client(*args, ollama_model="m")
        assert client.timeout == settings.LLM_TIMEOUT_SECONDS
        assert client.max_retries == 1
