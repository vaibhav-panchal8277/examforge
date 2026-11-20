"""
PDF text extraction.

Uploaded files are untrusted: the app calls extract_text_safely(), which checks
the file and parses it in a separate process (pdf_worker.py) with time and
memory limits, so a malicious or broken PDF can't freeze or crash the app.
"""
import itertools
import json
import os
import re
import subprocess
import sys

import pdfplumber
from pypdf import PdfReader

PDF_HEADER = b"%PDF-"
HEADER_SEARCH_BYTES = 1024  # the PDF spec allows the header anywhere in the first 1 KB
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(APP_DIR, "pdf_worker.py")

# Characters that are invisible when read but still reach the AI: zero-width
# characters, bidirectional overrides, soft hyphens, BOMs. A common way to hide
# injected instructions in a document.
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤⁦-⁩﻿]")
# Control characters other than tab and newline.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


class PDFError(ValueError):
    """The PDF can't be used; the message is safe to show to the user."""


def clean_text(text: str) -> str:
    return _CONTROL.sub("", _INVISIBLE.sub("", text))


def check_pdf_header(data: bytes) -> None:
    if PDF_HEADER not in data[:HEADER_SEARCH_BYTES]:
        raise PDFError("This file isn't a valid PDF. Please upload a PDF document.")


def _open_checked(pdf_file, max_pages):
    """Open with pypdf to reject password-protected or over-long PDFs before extracting."""
    try:
        reader = PdfReader(pdf_file)
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFError("This PDF is password-protected. Remove the password and upload it again.")
        page_count = len(reader.pages)
    except PDFError:
        raise
    except Exception as e:
        raise PDFError(f"Could not read the PDF: {e}") from e
    if max_pages and page_count > max_pages:
        raise PDFError(f"This PDF has {page_count} pages; the limit is {max_pages}. "
                       "Please upload just the syllabus pages.")
    return reader


def _is_memory_error(exc: BaseException) -> bool:
    """True if `exc` or anything that caused it is an out-of-memory error.
    (pdfminer wraps MemoryError as "Unable to allocate output buffer".)"""
    while exc is not None:
        if isinstance(exc, MemoryError) or "unable to allocate" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _collect(page_texts, max_chars) -> str:
    """Join page texts, stopping early once `max_chars` is reached (pages are read lazily)."""
    parts, total = [], 0
    for text in page_texts:
        if text:
            parts.append(text + "\n")
            total += len(text) + 1
        if max_chars and total >= max_chars:
            break
    return "".join(parts)


def _extract_with_pdfplumber(pdf_file, max_pages=None, max_chars=None) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        pages = itertools.islice(pdf.pages, max_pages) if max_pages else pdf.pages
        return _collect((page.extract_text() for page in pages), max_chars)


def _extract_with_pypdf(reader, max_pages=None, max_chars=None) -> str:
    pages = itertools.islice(reader.pages, max_pages) if max_pages else reader.pages
    return _collect((page.extract_text() for page in pages), max_chars)


def extract_text_from_pdf(pdf_file, max_pages=None, max_chars=None) -> str:
    """
    Extract text from a PDF file object (in this process; see extract_text_safely).

    Rejects password-protected PDFs and PDFs over `max_pages`, and stops after
    about `max_chars` characters. Tries pdfplumber first and falls back to pypdf
    if it fails or finds no text; each attempt builds its own result, so a
    failure partway through never leaves duplicated or partial text behind.

    Returns "" if the PDF opens but has no extractable text (e.g. scanned).
    Raises PDFError if the file can't be used.
    """
    reader = _open_checked(pdf_file, max_pages)
    pdf_file.seek(0)

    first_error = None
    try:
        text = _extract_with_pdfplumber(pdf_file, max_pages, max_chars)
        if text.strip():
            return clean_text(text)
    except Exception as e:
        if _is_memory_error(e):
            # Hit the worker's memory limit (e.g. a decompression bomb). Don't fall
            # back: pypdf would just silently return no text for the same file.
            raise PDFError("This PDF is too large or complex to read.") from e
        first_error = e

    try:
        return clean_text(_extract_with_pypdf(reader, max_pages, max_chars))
    except Exception as e:
        if first_error is None:
            # pdfplumber read the file fine; it just has no text.
            return ""
        raise PDFError(f"Could not read the PDF: {e}") from e


def _worker_env() -> dict:
    # Only what the interpreter needs: secrets such as OPENAI_API_KEY and
    # MONGODB_URI are deliberately not passed to the process parsing the upload.
    return {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "LANG") if k in os.environ}


def extract_text_safely(data: bytes, *, max_pages: int, max_chars: int,
                        timeout_seconds: int, memory_mb: int) -> str:
    """
    Check an uploaded PDF and extract its text in a separate, limited process.

    Raises PDFError with a user-friendly message if the file isn't a PDF, is
    password-protected, is too long, takes longer than `timeout_seconds`, or
    needs more than `memory_mb` of memory.
    """
    check_pdf_header(data)
    cmd = [sys.executable, WORKER, str(max_pages or 0), str(max_chars or 0),
           str(memory_mb), str(timeout_seconds)]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout_seconds,
                              cwd=APP_DIR, env=_worker_env())
    except subprocess.TimeoutExpired:
        raise PDFError("This PDF took too long to read. It may be damaged or unusually complex.") from None
    try:
        result = json.loads(proc.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        # No valid answer: the worker was killed, e.g. by the memory or CPU limit.
        raise PDFError("This PDF is too large or complex to read.") from None
    if not result.get("ok"):
        raise PDFError(result.get("error") or "Could not read the PDF.")
    return result["text"]
