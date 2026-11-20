import pytest
from collections import Counter

from distribution import distribute_questions

TOPICS = [{"topic_name": f"T{i}", "description": f"d{i}"} for i in range(3)]


def test_counts_and_marks_follow_distribution():
    plan = distribute_questions([{"marks": 5, "count": 4}, {"marks": 10, "count": 2}], TOPICS)
    assert Counter(q["marks"] for q in plan) == {5: 4, 10: 2}


def test_topics_are_not_repeated_until_all_used():
    plan = distribute_questions([{"marks": 5, "count": 3}], TOPICS)
    assert sorted(q["topic_name"] for q in plan) == ["T0", "T1", "T2"]


def test_cycles_when_more_questions_than_topics():
    plan = distribute_questions([{"marks": 2, "count": 7}], TOPICS)
    assert len(plan) == 7
    assert set(q["topic_name"] for q in plan) == {"T0", "T1", "T2"}


def test_no_topics_gives_empty_plan():
    assert distribute_questions([{"marks": 5, "count": 3}], []) == []


def test_caller_topics_list_is_not_mutated():
    topics = list(TOPICS)
    distribute_questions([{"marks": 5, "count": 5}], topics)
    assert topics == TOPICS


from distribution import validate_distribution


def test_valid_distribution():
    assert validate_distribution([{"marks": 5, "count": 4}, {"marks": 10, "count": 2}], 50, 100) == []


def test_zero_count_rows_are_ignored():
    assert validate_distribution([{"marks": 0, "count": 0}, {"marks": 5, "count": 1}], 50, 100) == []


@pytest.mark.parametrize("rows, message", [
    ([{"marks": 5, "count": -1}], "negative"),
    ([{"marks": 0, "count": 2}], "between 1 and 100"),
    ([{"marks": 500, "count": 1}], "between 1 and 100"),
    ([{"marks": 5, "count": 0}], "at least one question"),
    ([], "at least one question"),
    ([{"marks": 2, "count": 60}], "at most 50 questions"),
])
def test_invalid_distribution(rows, message):
    errors = validate_distribution(rows, 50, 100)
    assert any(message in e for e in errors), errors
