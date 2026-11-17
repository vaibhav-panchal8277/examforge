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
