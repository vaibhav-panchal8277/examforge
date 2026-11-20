"""
Parses one uploaded PDF in its own process (started by extractor.extract_text_safely).

Reads the PDF bytes from stdin and writes {"ok": true, "text": ...} or
{"ok": false, "error": ...} as JSON to stdout. Memory and CPU limits are applied
before any parsing, so a malicious PDF can only exhaust this process, which the
parent also kills after its timeout.

Usage: python pdf_worker.py MAX_PAGES MAX_CHARS MEMORY_MB CPU_SECONDS < file.pdf
"""
import io
import json
import sys


def _apply_limits(memory_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # Windows: only the parent's timeout applies
        return
    limits = [(resource.RLIMIT_AS, memory_mb * 1024 * 1024), (resource.RLIMIT_CPU, cpu_seconds)]
    for kind, value in limits:
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):  # not supported on this OS (e.g. RLIMIT_AS on macOS)
            pass


def main() -> None:
    max_pages, max_chars, memory_mb, cpu_seconds = (int(arg) for arg in sys.argv[1:5])
    data = sys.stdin.buffer.read()
    _apply_limits(memory_mb, cpu_seconds)

    from extractor import PDFError, extract_text_from_pdf

    try:
        text = extract_text_from_pdf(io.BytesIO(data), max_pages=max_pages or None,
                                     max_chars=max_chars or None)
        result = {"ok": True, "text": text}
    except PDFError as e:
        result = {"ok": False, "error": str(e)}
    except MemoryError:
        result = {"ok": False, "error": "This PDF is too large or complex to read."}
    except Exception:
        result = {"ok": False, "error": "Could not read the PDF."}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
