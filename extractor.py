import pdfplumber
from pypdf import PdfReader


def _join_pages(page_texts) -> str:
    return "".join(t + "\n" for t in page_texts if t)


def _extract_with_pdfplumber(pdf_file) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        return _join_pages(page.extract_text() for page in pdf.pages)


def _extract_with_pypdf(pdf_file) -> str:
    return _join_pages(page.extract_text() for page in PdfReader(pdf_file).pages)


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file.

    Tries pdfplumber first and falls back to pypdf if pdfplumber fails or finds
    no text. Each attempt builds its own result, so a failure partway through
    never leaves duplicated or partial text behind.

    Returns "" if the PDF opens but has no extractable text (e.g. scanned).
    Raises ValueError if neither library can read the file.
    """
    first_error = None
    try:
        text = _extract_with_pdfplumber(pdf_file)
        if text.strip():
            return text
    except Exception as e:
        first_error = e

    pdf_file.seek(0)
    try:
        return _extract_with_pypdf(pdf_file)
    except Exception as e:
        if first_error is None:
            # pdfplumber read the file fine; it just has no text.
            return ""
        raise ValueError(f"Could not read the PDF: {e}") from e
