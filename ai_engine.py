import json
from utils import get_llm_client, extract_json, create_json_completion

# Per-topic description length sent to the model; keeps prompts small.
MAX_DESCRIPTION_CHARS = 300


def generate_questions(
    question_plan: list,
    difficulty: str,
    use_openai: bool,
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "",
) -> list:
    """
    Generate exam questions from the question plan using the selected AI backend.
    Returns: [{'question_number', 'question_text', 'marks', 'topic_name'}]
    """
    client, model = get_llm_client(use_openai, openai_key, openai_model, ollama_model)

    # Compact plan to reduce tokens. The topic description (from the syllabus)
    # keeps questions tied to what the syllabus actually covers.
    compact_plan = []
    for q in question_plan:
        item = {"topic": q["topic_name"], "marks": q["marks"]}
        details = str(q.get("description") or "").strip()
        if details:
            item["covers"] = details[:MAX_DESCRIPTION_CHARS]
        compact_plan.append(item)

    prompt = (
        f"Generate {len(question_plan)} exam questions. Difficulty: {difficulty}. "
        "Low marks = short question. High marks = detailed question. "
        "Base each question on its topic and, when given, what the topic covers. "
        "Write exactly one question per item, in the same order as the list. "
        "Return ONLY JSON: {\"questions\": [{\"question_number\": 1, \"question_text\": \"...\", \"marks\": 5, \"topic_name\": \"...\"}]}. "
        "No markdown, no extra text.\n\n"
        f"Topics and marks:\n{json.dumps(compact_plan)}"
    )

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You are an exam question generator. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        # Scale with the number of questions so long papers aren't cut off mid-JSON.
        max_tokens=min(8192, max(2048, 200 * len(question_plan))),
        temperature=0.5,
    )

    response = create_json_completion(client, **kwargs)

    data = extract_json(response.choices[0].message.content)
    questions = data.get("questions", []) if isinstance(data, dict) else []
    return _merge_with_plan(question_plan, questions)


def _merge_with_plan(question_plan: list, questions: list) -> list:
    """
    Combine the model's question text with the configured plan.

    Marks and topics always come from the plan, never from the model, so the
    paper matches the configured distribution and marks are always integers.
    Raises ValueError if the model returned too few usable questions.
    """
    if not questions:
        raise ValueError("Model returned empty questions. Try reducing the number of questions or use a different model.")
    if len(questions) < len(question_plan):
        raise ValueError(
            f"Model returned {len(questions)} of {len(question_plan)} questions. "
            "Try again, reduce the number of questions, or use a different model."
        )

    result = []
    for i, (planned, q) in enumerate(zip(question_plan, questions), 1):
        text = q.get("question_text", "") if isinstance(q, dict) else q
        text = str(text or "").strip()
        if not text:
            raise ValueError(f"Model returned an empty question {i}. Try again or use a different model.")
        result.append({
            "question_number": i,
            "question_text": text,
            "marks": planned["marks"],
            "topic_name": planned["topic_name"],
        })
    return result
