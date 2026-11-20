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
    monkeypatch.setattr(extractor, "_extract_with_pdfplumber", lambda *a, **k: "")
    assert "Unit one" in extract_text_from_pdf(_make_pdf(["Unit one"]))


def test_unreadable_file_raises_value_error():
    with pytest.raises(ValueError, match="Could not read the PDF"):
        extract_text_from_pdf(io.BytesIO(b"this is not a pdf"))


# ── Untrusted uploads: extract_text_safely (separate, limited process) ───────
import sys
import zlib

from pypdf import PdfWriter

from extractor import PDFError, check_pdf_header, clean_text, extract_text_safely

LIMITS = dict(max_pages=50, max_chars=36000, timeout_seconds=30, memory_mb=512)


def _encrypted(user_password, owner_password=None) -> bytes:
    writer = PdfWriter(clone_from=_make_pdf(["Unit one"]))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _bomb(inflated_mb: int) -> bytes:
    """A one-page PDF whose content stream inflates from ~0.1% of its size to `inflated_mb`."""
    payload = zlib.compress(b"0 0 m " * (inflated_mb * 1024 * 1024 // 6), 9)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(payload) + payload + b"\nendstream"]
    pdf, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(pdf)


def test_safe_extraction_reads_normal_pdf():
    assert "Unit one" in extract_text_safely(_make_pdf(["Unit one", "Unit two"]).getvalue(), **LIMITS)


@pytest.mark.parametrize("data", [
    b"MZ\x90\x00 an executable renamed to .pdf",
    b"just some text",
    b"",
])
def test_non_pdf_is_rejected_before_parsing(data):
    with pytest.raises(PDFError, match="isn't a valid PDF"):
        extract_text_safely(data, **LIMITS)


def test_header_may_follow_leading_bytes():
    check_pdf_header(b"\x00" * 100 + b"%PDF-1.7 ...")  # allowed by the PDF spec


def test_password_protected_pdf_is_rejected():
    with pytest.raises(PDFError, match="password-protected"):
        extract_text_safely(_encrypted("secret"), **LIMITS)


def test_pdf_with_only_an_owner_password_is_read():
    assert "Unit one" in extract_text_safely(_encrypted("", owner_password="owner"), **LIMITS)


def test_too_many_pages_is_rejected():
    data = _make_pdf([f"Page {i}" for i in range(60)]).getvalue()
    with pytest.raises(PDFError, match="60 pages; the limit is 50"):
        extract_text_safely(data, **LIMITS)


def test_reading_stops_at_max_chars():
    data = _make_pdf([f"Page {i} " + "x" * 80 for i in range(40)]).getvalue()
    text = extract_text_safely(data, **{**LIMITS, "max_chars": 200})
    assert "Page 0" in text and "Page 39" not in text


def test_slow_pdf_times_out():
    with pytest.raises(PDFError, match="took too long"):
        extract_text_safely(_make_pdf(["Unit one"]).getvalue(), **{**LIMITS, "timeout_seconds": 0.01})


@pytest.mark.skipif(sys.platform == "win32", reason="memory limits need the resource module")
def test_pdf_bomb_is_stopped_by_memory_limit():
    data = _bomb(inflated_mb=256)
    assert len(data) < 1_000_000  # tiny file...
    with pytest.raises(PDFError, match="too large or complex"):
        extract_text_safely(data, **{**LIMITS, "memory_mb": 160})  # ...that needs far more memory


def test_worker_gets_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://user:pass@example")
    env = extractor._worker_env()
    assert "OPENAI_API_KEY" not in env and "MONGODB_URI" not in env


def test_clean_text_removes_hidden_characters():
    hidden = "Sort​ing‮ al­gorithms﻿\x07"
    assert clean_text(hidden) == "Sorting algorithms"
    assert clean_text("Line 1\n\tLine 2") == "Line 1\n\tLine 2"  # tabs and newlines kept
