from utils import get_llm_client, extract_json, create_json_completion


def parse_syllabus_structure(
    text: str,
    use_openai: bool,
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "",
) -> list:
    """
    Parse syllabus text and extract topic list using the selected AI backend.
    Returns: [{'topic_name': str, 'description': str}]
    """
    client, model = get_llm_client(use_openai, openai_key, openai_model, ollama_model)

    # Keep prompt short for local model speed
    prompt = (
        "Extract the main topics from this syllabus. "
        "Return ONLY a JSON object: {\"topics\": [{\"topic_name\": \"...\", \"description\": \"...\"}]}. "
        "No markdown, no extra text.\n\n"
        f"Syllabus (first 4000 chars):\n{text[:4000]}"
    )

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You extract syllabus topics. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    response = create_json_completion(client, **kwargs)

    data = extract_json(response.choices[0].message.content)
    topics = data.get("topics", [])
    if not topics:
        raise ValueError("Model returned empty topics. Try a different model or check the PDF.")
    return topics
