import json

from utils import get_llm_client, extract_json, create_json_completion

# Each request sees one chunk of the syllabus; small enough for local models.
CHUNK_CHARS = 4000
# Upper bound on requests per syllabus (~32k chars), to keep cost and time sane.
MAX_CHUNKS = 8


def split_text(text: str, size: int = CHUNK_CHARS) -> list:
    """Split text into chunks of at most `size` chars, breaking on line boundaries."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        # A single line longer than `size` is hard-split.
        while len(line) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(current) + len(line) > size:
            chunks.append(current)
            current = ""
        current += line
    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def is_truncated(text: str) -> bool:
    """True if the syllabus is longer than what parse_syllabus_structure reads."""
    return len(split_text(text)) > MAX_CHUNKS


def _clean_topics(raw) -> list:
    """Keep only well-formed topics: dicts with a non-empty topic_name."""
    topics = []
    for t in raw if isinstance(raw, list) else []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("topic_name") or "").strip()
        if name:
            topics.append({"topic_name": name, "description": str(t.get("description") or "").strip()})
    return topics


def parse_syllabus_structure(
    text: str,
    use_openai: bool,
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "",
) -> list:
    """
    Parse syllabus text and extract topic list using the selected AI backend.

    The text is split into chunks (up to MAX_CHUNKS) so topics from the whole
    syllabus are found, not just the beginning. Topics are de-duplicated by name.
    Returns: [{'topic_name': str, 'description': str}]
    """
    client, model = get_llm_client(use_openai, openai_key, openai_model, ollama_model)
    chunks = split_text(text)[:MAX_CHUNKS]

    topics, seen, parse_errors = [], set(), 0
    for i, chunk in enumerate(chunks, 1):
        prompt = (
            "Extract the main topics from this part of a syllabus. "
            "Return ONLY a JSON object: {\"topics\": [{\"topic_name\": \"...\", \"description\": \"...\"}]}. "
            "If this part has no syllabus topics (e.g. only references or admin details), return {\"topics\": []}. "
            "No markdown, no extra text.\n\n"
            f"Syllabus part {i} of {len(chunks)}:\n{chunk}"
        )
        response = create_json_completion(
            client,
            model=model,
            messages=[
                {"role": "system", "content": "You extract syllabus topics. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        # A chunk with unparseable output is skipped rather than failing the whole syllabus.
        try:
            data = extract_json(response.choices[0].message.content or "")
        except json.JSONDecodeError:
            parse_errors += 1
            continue

        for t in _clean_topics(data.get("topics") if isinstance(data, dict) else None):
            key = t["topic_name"].casefold()
            if key not in seen:
                seen.add(key)
                topics.append(t)

    if not topics:
        if parse_errors:
            raise ValueError("Model output could not be read as JSON. Try a different model.")
        raise ValueError("Model returned empty topics. Try a different model or check the PDF.")
    return topics
