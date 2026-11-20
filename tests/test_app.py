"""Runs the real app.py headlessly with Streamlit's AppTest; AI calls are faked."""
import os

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import ai_engine
import settings
import utils

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TOPICS = [{"topic_name": "Sorting", "description": ""}, {"topic_name": "Graphs", "description": ""}]


@pytest.fixture
def make_app(monkeypatch):
    monkeypatch.setattr(utils, "get_available_ollama_models", lambda: ["test-model"])
    monkeypatch.setattr(settings, "ACCESS_PASSWORD", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    calls = []

    def fake_generate(plan, difficulty, **kw):
        calls.append(kw)
        return ai_engine._merge_with_plan(plan, [{"question_text": f"Q{i}"} for i in range(len(plan))])

    monkeypatch.setattr(ai_engine, "generate_questions", fake_generate)
    st.cache_resource.clear()  # fresh app-wide rate limiter per test

    def make():
        at = AppTest.from_file(APP, default_timeout=30)
        at.calls = calls
        return at

    yield make
    st.cache_resource.clear()


def _generate(at):
    [b for b in at.button if b.label == "Generate Questions"][0].click().run()


def _messages(elements):
    return [e.value for e in elements]


def test_no_password_shows_app(make_app):
    at = make_app().run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_password_gate(make_app, monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_PASSWORD", "s3cret")
    at = make_app().run()
    assert len(at.tabs) == 0  # nothing but the sign-in form
    assert "password protected" in " ".join(_messages(at.caption))

    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert "Incorrect password." in _messages(at.error)
    assert len(at.tabs) == 0

    at.text_input[0].input("s3cret")
    at.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_usage_limit_blocks_extra_requests(make_app, monkeypatch):
    monkeypatch.setattr(settings, "MAX_AI_REQUESTS_PER_HOUR", 1)
    at = make_app()
    at.session_state["topics"] = TOPICS
    at.run()
    _generate(at)
    assert len(at.calls) == 1
    _generate(at)
    assert len(at.calls) == 1  # second request refused
    assert any("limit of 1 AI requests per hour" in m for m in _messages(at.error))


def test_own_openai_key_is_not_limited(make_app, monkeypatch):
    monkeypatch.setattr(settings, "MAX_AI_REQUESTS_PER_HOUR", 1)
    at = make_app()
    at.session_state["topics"] = TOPICS
    at.session_state["use_openai"] = True
    at.session_state["openai_key"] = "sk-visitor-test"
    at.run()
    for _ in range(3):
        _generate(at)
    assert len(at.calls) == 3
    assert all(c["openai_key"] == "sk-visitor-test" for c in at.calls)


def test_invalid_marks_table_is_rejected(make_app):
    at = make_app()
    at.session_state["topics"] = TOPICS
    at.session_state["marks_distribution"] = pd.DataFrame({"Marks": [2], "Questions": [500]})
    at.run()
    _generate(at)
    assert at.calls == []
    assert any("at most 50 questions" in m for m in _messages(at.error))
