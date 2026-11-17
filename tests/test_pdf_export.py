import pdfplumber
import pytest

from pdf_export import generate_pdf


def _pdf_text(buffer) -> str:
    with pdfplumber.open(buffer) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


@pytest.mark.parametrize("question", [
    "What does vector<int> store?",
    "Compare List<String> and Map<K,V> in Java",
    "Is 3<5 && 7>2?",
])
def test_angle_brackets_are_kept(question):
    text = _pdf_text(generate_pdf("Exam", "CS", [{"question_text": question, "marks": 5}]))
    assert question in text


@pytest.mark.parametrize("question", [
    "Use the <br> and <p> tags",
    "Explain a<b>c ordering",
    "Write HTML for <b>bold</b> and <i>italic",
    "Use <font size=x> tag",
])
def test_markup_like_text_does_not_crash(question):
    text = _pdf_text(generate_pdf("Exam", "CS", [{"question_text": question, "marks": 5}]))
    assert question in text


def test_header_and_total_marks():
    questions = [{"question_text": "Q one", "marks": 5}, {"question_text": "Q two", "marks": 10}]
    text = _pdf_text(generate_pdf("Midterm & <Final>", "R&D", questions))
    assert "Midterm & <Final>" in text
    assert "R&D" in text
    assert "Total Marks: 15" in text
