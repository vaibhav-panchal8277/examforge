from types import SimpleNamespace

import pytest

import ai_engine
from ai_engine import _merge_with_plan

PLAN = [
    {"topic_name": "Sorting", "description": "", "marks": 5},
    {"topic_name": "Graphs", "description": "", "marks": 10},
]


def test_marks_and_topics_come_from_plan():
    model_output = [
        {"question_text": "Explain quicksort.", "marks": "50", "topic_name": "Wrong"},
        {"question_text": "Explain BFS.", "marks": 1, "topic_name": "Also wrong"},
    ]
    result = _merge_with_plan(PLAN, model_output)
    assert [q["marks"] for q in result] == [5, 10]
    assert [q["topic_name"] for q in result] == ["Sorting", "Graphs"]
    assert [q["question_number"] for q in result] == [1, 2]
    assert sum(q["marks"] for q in result) == 15


def test_extra_questions_are_dropped():
    model_output = [{"question_text": f"Q{i}"} for i in range(5)]
    assert len(_merge_with_plan(PLAN, model_output)) == len(PLAN)


def test_too_few_questions_raises():
    with pytest.raises(ValueError, match="1 of 2"):
        _merge_with_plan(PLAN, [{"question_text": "Only one"}])


def test_empty_questions_raises():
    with pytest.raises(ValueError, match="empty"):
        _merge_with_plan(PLAN, [])


def test_blank_question_text_raises():
    with pytest.raises(ValueError, match="empty question 2"):
        _merge_with_plan(PLAN, [{"question_text": "ok"}, {"question_text": "   "}])


def test_plain_string_items_are_accepted():
    result = _merge_with_plan(PLAN, ["Explain quicksort.", "Explain BFS."])
    assert result[1]["question_text"] == "Explain BFS."


def test_generate_questions_end_to_end(monkeypatch):
    content = '```json\n{"questions": [{"question_text": "A", "marks": "5"}, {"question_text": "B"}]}\n```'
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(ai_engine, "get_llm_client", lambda *a, **k: (client, "test-model"))

    result = ai_engine.generate_questions(PLAN, "Medium", use_openai=False, ollama_model="test-model")
    assert [(q["question_text"], q["marks"]) for q in result] == [("A", 5), ("B", 10)]
