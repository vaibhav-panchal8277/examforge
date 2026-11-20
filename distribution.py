import random

def distribute_questions(marks_distribution: list, topics: list) -> list:
    """
    marks_distribution: list of dicts [{'marks': 5, 'count': 4}, {'marks': 8, 'count': 2}]
    topics: list of dicts [{'topic_name': '...', 'description': '...'}]
    
    Returns a list of question requirements:
    [{'topic_name': '...', 'description': '...', 'marks': 5}, ...]
    """
    question_plan = []
    
    if not topics:
        return question_plan
        
    topic_pool = topics.copy()
    
    for dist in marks_distribution:
        marks = dist.get('marks', 0)
        count = dist.get('count', 0)
        
        for _ in range(count):
            if not topic_pool:
                topic_pool = topics.copy()
            
            selected_topic = random.choice(topic_pool)
            topic_pool.remove(selected_topic)
            
            question_plan.append({
                'topic_name': selected_topic.get('topic_name', 'Unknown Topic'),
                'description': selected_topic.get('description', ''),
                'marks': marks
            })
            
    return question_plan


def validate_distribution(marks_distribution: list, max_questions: int, max_marks: int) -> list:
    """
    Check the marks table before generating. Returns a list of error messages
    (empty if valid). Rows with a count of 0 are ignored.
    """
    errors = []
    for row in marks_distribution:
        marks, count = row.get('marks', 0), row.get('count', 0)
        if count < 0:
            errors.append(f"Question counts can't be negative (row with {marks} marks).")
        elif count > 0 and not 1 <= marks <= max_marks:
            errors.append(f"Marks per question must be between 1 and {max_marks} (got {marks}).")

    total = sum(max(0, row.get('count', 0)) for row in marks_distribution)
    if total == 0:
        errors.append("Add at least one question to the marks table.")
    elif total > max_questions:
        errors.append(f"A paper can have at most {max_questions} questions (the table asks for {total}).")
    return errors
