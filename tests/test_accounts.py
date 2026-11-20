import os
import uuid

import mongomock
import pytest
from pymongo import MongoClient

import accounts
from accounts import AccountError, Store

QUESTIONS = [
    {"question_number": 1, "question_text": "Explain <b> & vector<int>.", "marks": 5, "topic_name": "Basics"},
    {"question_number": 2, "question_text": "Describe pipelining.", "marks": 10, "topic_name": "CPU"},
]

BACKENDS = ["mongomock"] + (["mongodb"] if os.getenv("MONGODB_TEST_URI") else [])


@pytest.fixture(params=BACKENDS)
def store(request):
    """A fresh, empty database: in-memory always, plus a real MongoDB when MONGODB_TEST_URI is set."""
    if request.param == "mongomock":
        yield Store(mongomock.MongoClient()["examforge_test"])
        return
    client = MongoClient(os.environ["MONGODB_TEST_URI"], serverSelectionTimeoutMS=5000)
    name = f"examforge_test_{uuid.uuid4().hex[:8]}"
    yield Store(client[name])
    client.drop_database(name)


def test_create_and_authenticate(store):
    user = store.create_user("  Teacher@School.EDU ", "correct horse battery")
    assert user["email"] == "teacher@school.edu"
    assert store.authenticate("teacher@school.edu", "correct horse battery") == user
    assert store.authenticate("TEACHER@school.edu", "correct horse battery") == user  # case-insensitive


def test_wrong_password_and_unknown_email(store):
    store.create_user("a@b.co", "password123")
    assert store.authenticate("a@b.co", "password124") is None
    assert store.authenticate("nobody@b.co", "password123") is None
    assert store.authenticate("", "password123") is None
    assert store.authenticate("a@b.co", "x" * 1000) is None


def test_password_is_hashed_not_stored(store):
    store.create_user("a@b.co", "password123")
    doc = store.users.find_one({"email": "a@b.co"})
    assert "password123" not in str(doc)
    assert doc["password_hash"].startswith("$argon2id$")


@pytest.mark.parametrize("email, password, message", [
    ("not-an-email", "password123", "valid email"),
    ("a@b", "password123", "valid email"),
    ("a@b.co", "short", "at least 8"),
    ("a@b.co", "x" * 129, "at most 128"),
])
def test_invalid_sign_up(store, email, password, message):
    with pytest.raises(AccountError, match=message):
        store.create_user(email, password)


def test_duplicate_email_rejected(store):
    store.create_user("a@b.co", "password123")
    with pytest.raises(AccountError, match="already exists"):
        store.create_user("A@B.CO", "different-pass")


def test_save_list_get_paper(store):
    uid = store.create_user("a@b.co", "password123")["id"]
    pid = store.save_paper(uid, "Midterm <Test> & Quiz", "CS", "Hard", QUESTIONS)
    papers = store.list_papers(uid)
    assert [p["id"] for p in papers] == [pid]
    paper = store.get_paper(uid, pid)
    assert paper["exam_name"] == "Midterm <Test> & Quiz"
    assert paper["total_marks"] == 15
    assert paper["questions"][0]["question_text"] == "Explain <b> & vector<int>."
    assert "created_at" in paper and "_id" not in paper


def test_papers_newest_first(store):
    uid = store.create_user("a@b.co", "password123")["id"]
    first = store.save_paper(uid, "First", "CS", "Easy", QUESTIONS)
    second = store.save_paper(uid, "Second", "CS", "Easy", QUESTIONS)
    assert [p["id"] for p in store.list_papers(uid)] == [second, first]


def test_users_cannot_see_or_delete_each_others_papers(store):
    alice = store.create_user("alice@b.co", "password123")["id"]
    bob = store.create_user("bob@b.co", "password123")["id"]
    pid = store.save_paper(alice, "Alice's paper", "CS", "Easy", QUESTIONS)

    assert store.list_papers(bob) == []
    assert store.get_paper(bob, pid) is None
    assert store.delete_paper(bob, pid) is False
    assert store.get_paper(alice, pid) is not None

    assert store.delete_paper(alice, pid) is True
    assert store.list_papers(alice) == []


@pytest.mark.parametrize("bad_id", ["not-an-id", "", None, "0" * 24])
def test_bad_paper_ids_are_handled(store, bad_id):
    uid = store.create_user("a@b.co", "password123")["id"]
    assert store.get_paper(uid, bad_id) is None
    assert store.delete_paper(uid, bad_id) is False


def test_connect_fails_fast_when_unreachable():
    with pytest.raises(Exception):
        accounts.connect("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=500", "x")


# ── "Stay signed in" sessions ────────────────────────────────────────────────
from datetime import datetime, timedelta, timezone


def test_session_restores_user(store):
    user = store.create_user("a@b.co", "password123")
    token = store.create_session(user["id"], days=7)
    assert len(token) >= 40
    assert store.user_for_session(token) == user


def test_only_a_hash_of_the_token_is_stored(store):
    user = store.create_user("a@b.co", "password123")
    token = store.create_session(user["id"], days=7)
    doc = store.sessions.find_one({})
    assert token not in str(doc) and len(doc["token_hash"]) == 64


def test_expired_session_is_rejected(store):
    user = store.create_user("a@b.co", "password123")
    token = store.create_session(user["id"], days=7)
    store.sessions.update_many({}, {"$set": {"expires_at": datetime.now(timezone.utc) - timedelta(minutes=1)}})
    assert store.user_for_session(token) is None


def test_signed_out_session_is_rejected(store):
    user = store.create_user("a@b.co", "password123")
    token = store.create_session(user["id"], days=7)
    store.delete_session(token)
    assert store.user_for_session(token) is None


@pytest.mark.parametrize("token", ["", None, "made-up-token", "x" * 500])
def test_unknown_or_odd_tokens_are_rejected(store, token):
    assert store.user_for_session(token) is None
    store.delete_session(token)  # must not raise


def test_session_for_deleted_user_is_rejected(store):
    user = store.create_user("a@b.co", "password123")
    token = store.create_session(user["id"], days=7)
    store.users.delete_many({})
    assert store.user_for_session(token) is None


def test_sessions_are_per_user(store):
    alice = store.create_user("alice@b.co", "password123")
    bob = store.create_user("bob@b.co", "password123")
    ta, tb = store.create_session(alice["id"], 7), store.create_session(bob["id"], 7)
    assert store.user_for_session(ta)["email"] == "alice@b.co"
    assert store.user_for_session(tb)["email"] == "bob@b.co"
