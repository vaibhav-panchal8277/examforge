import io

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import extractor
from extractor import extract_text_from_pdf


def _make_pdf(pages) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        if text:
            c.drawString(72, 700, text)
        c.showPage()
    c.save()
    buf.seek(0)
    return buf


def test_extracts_all_pages():
    text = extract_text_from_pdf(_make_pdf(["Unit one", "Unit two"]))
    assert "Unit one" in text and "Unit two" in text


def test_pdf_without_text_returns_empty():
    assert extract_text_from_pdf(_make_pdf([None])).strip() == ""


def test_pdfplumber_failing_midway_does_not_duplicate(monkeypatch):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            if self.text is None:
                raise RuntimeError("broken page")
            return self.text

    class FakePDF:
        pages = [Page("Unit one"), Page(None)]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(extractor.pdfplumber, "open", lambda f: FakePDF())
    text = extract_text_from_pdf(_make_pdf(["Unit one", "Unit two"]))
    # Falls back to pypdf, which reads both pages exactly once.
    assert text.count("Unit one") == 1
    assert "Unit two" in text


def test_falls_back_when_pdfplumber_finds_no_text(monkeypatch):
    monkeypatch.setattr(extractor, "_extract_with_pdfplumber", lambda f: "")
    assert "Unit one" in extract_text_from_pdf(_make_pdf(["Unit one"]))


def test_unreadable_file_raises_value_error():
    with pytest.raises(ValueError, match="Could not read the PDF"):
        extract_text_from_pdf(io.BytesIO(b"this is not a pdf"))
