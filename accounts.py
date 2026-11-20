"""
User accounts and saved papers, stored in MongoDB.

Collections:
  users:  {_id, email (unique, lowercase), password_hash, created_at}
  papers: {_id, user_id, exam_name, subject_name, difficulty, total_marks,
           questions: [{question_number, question_text, marks, topic_name}], created_at}

Every paper query is filtered by user_id, so a user can only ever read or
delete their own papers.
"""
import re
from datetime import datetime, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128  # bounds hashing work per sign-in attempt
MAX_EMAIL_LENGTH = 254
MAX_PAPERS_LISTED = 50
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_hasher = PasswordHasher()  # Argon2id with the library's recommended parameters
# Verified against when an email doesn't exist, so "no such user" and
# "wrong password" take the same time and can't be told apart.
_DUMMY_HASH = _hasher.hash("not-a-real-password")


class AccountError(ValueError):
    """A problem the user can fix (bad email, weak password, email taken)."""


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _object_id(value):
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


class Store:
    def __init__(self, db):
        self.users = db["users"]
        self.papers = db["papers"]
        self.users.create_index("email", unique=True)
        self.papers.create_index([("user_id", 1), ("created_at", DESCENDING), ("_id", DESCENDING)])

    # ── Users ────────────────────────────────────────────────────────────────
    def create_user(self, email: str, password: str) -> dict:
        email = normalize_email(email)
        if len(email) > MAX_EMAIL_LENGTH or not EMAIL_RE.match(email):
            raise AccountError("Enter a valid email address.")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if len(password) > MAX_PASSWORD_LENGTH:
            raise AccountError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
        doc = {"email": email, "password_hash": _hasher.hash(password), "created_at": _now()}
        try:
            result = self.users.insert_one(doc)
        except DuplicateKeyError:
            raise AccountError("An account with this email already exists. Sign in instead.") from None
        return {"id": str(result.inserted_id), "email": email}

    def authenticate(self, email: str, password: str):
        """Return {"id", "email"} if the email and password match, else None."""
        email = normalize_email(email)
        if not email or len(password) > MAX_PASSWORD_LENGTH:
            return None
        user = self.users.find_one({"email": email})
        try:
            _hasher.verify(user["password_hash"] if user else _DUMMY_HASH, password)
        except (VerifyMismatchError, InvalidHashError):
            return None
        if user is None:
            return None
        if _hasher.check_needs_rehash(user["password_hash"]):
            self.users.update_one({"_id": user["_id"]},
                                  {"$set": {"password_hash": _hasher.hash(password)}})
        return {"id": str(user["_id"]), "email": user["email"]}

    # ── Papers ───────────────────────────────────────────────────────────────
    def save_paper(self, user_id: str, exam_name: str, subject_name: str,
                   difficulty: str, questions: list) -> str:
        doc = {
            "user_id": user_id,
            "exam_name": str(exam_name)[:200],
            "subject_name": str(subject_name)[:200],
            "difficulty": str(difficulty)[:20],
            "total_marks": sum(int(q.get("marks", 0)) for q in questions),
            "questions": [
                {
                    "question_number": int(q.get("question_number", i)),
                    "question_text": str(q.get("question_text", "")),
                    "marks": int(q.get("marks", 0)),
                    "topic_name": str(q.get("topic_name", "")),
                }
                for i, q in enumerate(questions, 1)
            ],
            "created_at": _now(),
        }
        return str(self.papers.insert_one(doc).inserted_id)

    def list_papers(self, user_id: str, limit: int = MAX_PAPERS_LISTED) -> list:
        # _id breaks ties: MongoDB stores times to the millisecond, and ObjectIds always increase.
        cursor = (self.papers.find({"user_id": user_id})
                  .sort([("created_at", DESCENDING), ("_id", DESCENDING)]).limit(limit))
        return [self._paper_out(doc) for doc in cursor]

    def get_paper(self, user_id: str, paper_id: str):
        oid = _object_id(paper_id)
        doc = self.papers.find_one({"_id": oid, "user_id": user_id}) if oid else None
        return self._paper_out(doc) if doc else None

    def delete_paper(self, user_id: str, paper_id: str) -> bool:
        oid = _object_id(paper_id)
        if not oid:
            return False
        return self.papers.delete_one({"_id": oid, "user_id": user_id}).deleted_count == 1

    @staticmethod
    def _paper_out(doc: dict) -> dict:
        out = {k: v for k, v in doc.items() if k != "_id"}
        out["id"] = str(doc["_id"])
        return out


def connect(uri: str, db_name: str) -> Store:
    """Connect to MongoDB and fail fast (5s) with a clear error if it's unreachable."""
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="examforge")
    client.admin.command("ping")
    return Store(client[db_name])
