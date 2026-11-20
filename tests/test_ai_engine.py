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


def _capture_prompt(monkeypatch, n):
    seen = {}
    content = '{"questions": [%s]}' % ", ".join('{"question_text": "Q%d"}' % i for i in range(n))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def create(**kw):
        seen.update(kw)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(ai_engine, "get_llm_client", lambda *a, **k: (client, "m"))
    return seen


def test_topic_descriptions_are_sent(monkeypatch):
    seen = _capture_prompt(monkeypatch, 2)
    plan = [
        {"topic_name": "Sorting", "description": "Quicksort and mergesort analysis", "marks": 5},
        {"topic_name": "Graphs", "description": "", "marks": 10},
    ]
    ai_engine.generate_questions(plan, "Hard", use_openai=False, ollama_model="m")
    prompt = seen["messages"][-1]["content"]
    assert "Quicksort and mergesort analysis" in prompt
    assert '"topic": "Graphs", "marks": 10}' in prompt  # no empty "covers" key


def test_long_descriptions_are_trimmed(monkeypatch):
    seen = _capture_prompt(monkeypatch, 1)
    plan = [{"topic_name": "T", "description": "d" * 5000, "marks": 5}]
    ai_engine.generate_questions(plan, "Easy", use_openai=False, ollama_model="m")
    prompt = seen["messages"][-1]["content"]
    assert "d" * ai_engine.MAX_DESCRIPTION_CHARS in prompt
    assert "d" * (ai_engine.MAX_DESCRIPTION_CHARS + 1) not in prompt


def test_max_tokens_scales_with_question_count(monkeypatch):
    seen = _capture_prompt(monkeypatch, 30)
    plan = [{"topic_name": "T", "description": "", "marks": 5}] * 30
    ai_engine.generate_questions(plan, "Easy", use_openai=False, ollama_model="m")
    assert seen["max_tokens"] == 6000


def test_question_text_is_cleaned_and_capped():
    long_hidden = "Explain​ sorting " + "z" * 5000
    result = _merge_with_plan(PLAN, [{"question_text": long_hidden}, {"question_text": "Q2"}])
    assert result[0]["question_text"].startswith("Explain sorting ")
    assert len(result[0]["question_text"]) <= ai_engine.MAX_QUESTION_CHARS


def test_generation_prompt_marks_topics_untrusted(monkeypatch):
    seen = _capture_prompt(monkeypatch, 1)
    ai_engine.generate_questions([{"topic_name": "T", "description": "", "marks": 5}], "Easy",
                                 use_openai=False, ollama_model="m")
    assert "untrusted" in seen["messages"][0]["content"]
