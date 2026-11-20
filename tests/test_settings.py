import importlib

import pytest

import settings


@pytest.fixture
def reload_settings(monkeypatch):
    for name in ["OLLAMA_BASE_URL", "OLLAMA_HOST", "EXAMFORGE_MAX_AI_REQUESTS_PER_HOUR",
                 "EXAMFORGE_LLM_TIMEOUT", "EXAMFORGE_PASSWORD", "MONGODB_URI", "MONGODB_DB"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore any local .env
    yield lambda: importlib.reload(settings)
    monkeypatch.undo()
    importlib.reload(settings)


def test_defaults(reload_settings):
    s = reload_settings()
    assert s.OLLAMA_BASE_URL == "http://localhost:11434"
    assert s.MAX_AI_REQUESTS_PER_HOUR == 120
    assert s.LLM_TIMEOUT_SECONDS == 120
    assert s.ACCESS_PASSWORD == ""
    assert s.MONGODB_URI == ""
    assert s.MONGODB_DB == "examforge"
    assert s.MAX_AI_REQUESTS_PER_USER_PER_HOUR == 30


@pytest.mark.parametrize("env, expected", [
    ({"OLLAMA_BASE_URL": "http://ollama:11434/"}, "http://ollama:11434"),
    ({"OLLAMA_HOST": "10.0.0.5:11434"}, "http://10.0.0.5:11434"),
    ({"OLLAMA_HOST": "https://ollama.example.com"}, "https://ollama.example.com"),
    ({"OLLAMA_BASE_URL": "http://a:1", "OLLAMA_HOST": "b:2"}, "http://a:1"),
])
def test_ollama_url(reload_settings, monkeypatch, env, expected):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert reload_settings().OLLAMA_BASE_URL == expected


def test_integer_settings(reload_settings, monkeypatch):
    monkeypatch.setenv("EXAMFORGE_MAX_AI_REQUESTS_PER_HOUR", "0")
    monkeypatch.setenv("EXAMFORGE_LLM_TIMEOUT", "0")  # clamped to at least 1
    s = reload_settings()
    assert s.MAX_AI_REQUESTS_PER_HOUR == 0
    assert s.LLM_TIMEOUT_SECONDS == 1


def test_bad_integer_gives_clear_error(reload_settings, monkeypatch):
    monkeypatch.setenv("EXAMFORGE_LLM_TIMEOUT", "two minutes")
    with pytest.raises(ValueError, match="EXAMFORGE_LLM_TIMEOUT"):
        reload_settings()
