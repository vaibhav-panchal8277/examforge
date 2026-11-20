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
    monkeypatch.setattr(settings, "MONGODB_URI", "")  # accounts off unless a test turns them on
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


# ── Accounts mode (MONGODB_URI set; an in-memory MongoDB stands in) ──────────
import mongomock

import accounts


@pytest.fixture
def make_accounts_app(make_app, monkeypatch):
    shared_store = accounts.Store(mongomock.MongoClient()["examforge_test"])
    monkeypatch.setattr(settings, "MONGODB_URI", "mongodb://in-memory-for-tests")
    monkeypatch.setattr(accounts, "connect", lambda uri, db: shared_store)

    def make():
        at = make_app()
        at.store = shared_store
        return at

    return make


def _click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def _sign_up(at, email, password="password123", confirm=None):
    at.text_input(key="signup_email").input(email)
    at.text_input(key="signup_password").input(password)
    at.text_input(key="signup_confirm").input(confirm or password)
    _click(at, "Create account")


def _sign_in(at, email, password="password123"):
    at.text_input(key="signin_email").input(email)
    at.text_input(key="signin_password").input(password)
    _click(at, "Sign in")


def test_accounts_require_sign_in(make_accounts_app):
    at = make_accounts_app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Sign in", "Create account"]  # only the account screen


def test_sign_up_then_app_shows_my_papers(make_accounts_app):
    at = make_accounts_app().run()
    _sign_up(at, "teacher@school.edu")
    assert not at.exception
    assert [t.label for t in at.tabs][-1] == "My Papers"
    assert any("teacher@school.edu" in m.value for m in at.markdown)


@pytest.mark.parametrize("kwargs, message", [
    ({"confirm": "different123"}, "Passwords don't match."),
    ({"password": "short"}, "at least 8"),
])
def test_sign_up_errors(make_accounts_app, kwargs, message):
    at = make_accounts_app().run()
    _sign_up(at, "teacher@school.edu", **kwargs)
    assert any(message in m for m in _messages(at.error))
    assert "My Papers" not in [t.label for t in at.tabs]


def test_sign_in_wrong_then_right_password(make_accounts_app):
    at = make_accounts_app()
    at.store.create_user("teacher@school.edu", "password123")
    at.run()
    _sign_in(at, "teacher@school.edu", "wrong-password")
    assert "Incorrect email or password." in _messages(at.error)
    _sign_in(at, "Teacher@School.edu", "password123")
    assert "My Papers" in [t.label for t in at.tabs]


def test_generated_papers_are_saved_per_user(make_accounts_app):
    alice = make_accounts_app().run()
    _sign_up(alice, "alice@school.edu")
    alice.session_state["topics"] = TOPICS
    alice.run()
    _generate(alice)
    assert any("saved it to **My Papers**" in m for m in _messages(alice.success))
    alice_id = alice.session_state["user"]["id"]
    assert len(alice.store.list_papers(alice_id)) == 1

    bob = make_accounts_app().run()  # a different browser session
    _sign_up(bob, "bob@school.edu")
    assert bob.store.list_papers(bob.session_state["user"]["id"]) == []
    assert any("saved here automatically" in i.value for i in bob.info)


def test_sign_out_clears_everything(make_accounts_app):
    at = make_accounts_app().run()
    _sign_up(at, "teacher@school.edu")
    at.session_state["topics"] = TOPICS
    at.run()
    _click(at, "Sign out")
    assert [t.label for t in at.tabs] == ["Sign in", "Create account"]
    assert "user" not in at.session_state
    assert "topics" not in at.session_state


def test_per_user_limit(make_accounts_app, monkeypatch):
    monkeypatch.setattr(settings, "MAX_AI_REQUESTS_PER_USER_PER_HOUR", 1)
    at = make_accounts_app().run()
    _sign_up(at, "teacher@school.edu")
    at.session_state["topics"] = TOPICS
    at.run()
    _generate(at)
    _generate(at)
    assert len(at.calls) == 1
    assert any("You've reached your limit of 1 AI requests per hour" in m for m in _messages(at.error))


def test_database_unreachable_shows_friendly_error(make_app, monkeypatch):
    monkeypatch.setattr(settings, "MONGODB_URI", "mongodb://unreachable")

    def boom(uri, db):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(accounts, "connect", boom)
    at = make_app().run()
    assert not at.exception
    assert any("can't reach its database" in m for m in _messages(at.error))


def test_non_pdf_upload_shows_clear_error(make_app):
    at = make_app().run()
    at.file_uploader[0].upload("syllabus.pdf", b"MZ\x90\x00 not really a pdf", "application/pdf").run()
    _click(at, "Extract & Analyze Syllabus")
    assert not at.exception
    assert any("isn't a valid PDF" in m for m in _messages(at.error))
    assert at.session_state["topics"] == []
