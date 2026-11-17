from types import SimpleNamespace

import pytest

import syllabus_parser
from syllabus_parser import MAX_CHUNKS, is_truncated, parse_syllabus_structure, split_text


def test_split_text_respects_size_and_keeps_all_text():
    text = "".join(f"Line {i}: some syllabus content here\n" for i in range(500))
    chunks = split_text(text, size=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text


def test_split_text_breaks_on_lines():
    chunks = split_text("aaaa\nbbbb\ncccc\n", size=10)
    assert chunks == ["aaaa\nbbbb\n", "cccc\n"]


def test_split_text_hard_splits_long_lines():
    chunks = split_text("x" * 25, size=10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_split_text_drops_blank_chunks():
    assert split_text("\n\n   \n") == []


def test_is_truncated():
    assert not is_truncated("short")
    assert is_truncated("y" * (syllabus_parser.CHUNK_CHARS * (MAX_CHUNKS + 1)))


class FakeClient:
    """Returns one canned response per call, in order, and records prompts."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        content = self.contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _use(monkeypatch, client):
    monkeypatch.setattr(syllabus_parser, "get_llm_client", lambda *a, **k: (client, "m"))


def _long_syllabus(parts):
    size = syllabus_parser.CHUNK_CHARS
    return "".join(f"UNIT {i}\n" + "z" * (size - 10) + "\n" for i in range(parts))


def test_reads_whole_syllabus_and_dedupes(monkeypatch):
    client = FakeClient([
        '{"topics": [{"topic_name": "Sorting", "description": "quick, merge"}]}',
        '{"topics": [{"topic_name": "sorting ", "description": "dup"}, {"topic_name": "Graphs"}]}',
        '{"topics": [{"topic_name": "Trees", "description": "BST"}]}',
    ])
    _use(monkeypatch, client)
    topics = parse_syllabus_structure(_long_syllabus(3), use_openai=False, ollama_model="m")
    assert len(client.prompts) == 3
    assert "UNIT 2" in client.prompts[2]  # the end of the syllabus is sent too
    assert topics == [
        {"topic_name": "Sorting", "description": "quick, merge"},
        {"topic_name": "Graphs", "description": ""},
        {"topic_name": "Trees", "description": "BST"},
    ]


def test_caps_number_of_requests(monkeypatch):
    client = FakeClient(['{"topics": [{"topic_name": "T%d"}]}' % i for i in range(MAX_CHUNKS + 5)])
    _use(monkeypatch, client)
    topics = parse_syllabus_structure(_long_syllabus(MAX_CHUNKS + 5), use_openai=False, ollama_model="m")
    assert len(client.prompts) == MAX_CHUNKS
    assert len(topics) == MAX_CHUNKS


def test_bad_chunk_is_skipped(monkeypatch):
    client = FakeClient(["not json at all", '{"topics": [{"topic_name": "Graphs"}]}'])
    _use(monkeypatch, client)
    topics = parse_syllabus_structure(_long_syllabus(2), use_openai=False, ollama_model="m")
    assert [t["topic_name"] for t in topics] == ["Graphs"]


def test_malformed_topics_are_dropped(monkeypatch):
    client = FakeClient(['{"topics": ["just a string", {"description": "no name"}, {"topic_name": " OK "}]}'])
    _use(monkeypatch, client)
    topics = parse_syllabus_structure("Unit 1: OK", use_openai=False, ollama_model="m")
    assert topics == [{"topic_name": "OK", "description": ""}]


def test_all_chunks_unreadable_raises(monkeypatch):
    _use(monkeypatch, FakeClient(["nope"]))
    with pytest.raises(ValueError, match="JSON"):
        parse_syllabus_structure("Unit 1", use_openai=False, ollama_model="m")


def test_no_topics_raises(monkeypatch):
    _use(monkeypatch, FakeClient(['{"topics": []}']))
    with pytest.raises(ValueError, match="empty topics"):
        parse_syllabus_structure("Unit 1", use_openai=False, ollama_model="m")
