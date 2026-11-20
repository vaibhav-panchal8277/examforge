"""
App settings, read once from environment variables (or a local .env file).

Real environment variables win over .env, so a deployment's configuration
can't be overridden by a stray .env file. See .env.example for all options.
"""
import os

from dotenv import load_dotenv

load_dotenv(override=False)


def _int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a whole number, got {raw!r}") from None


def _ollama_url() -> str:
    # OLLAMA_BASE_URL wins; OLLAMA_HOST is Ollama's own variable ("host:port" or a URL).
    url = os.getenv("OLLAMA_BASE_URL", "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    if not url:
        return "http://localhost:11434"
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


# OpenAI key used when a visitor doesn't enter their own. Server-side only.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Where the Ollama server runs (for Docker, e.g. http://host.docker.internal:11434).
OLLAMA_BASE_URL = _ollama_url()

# Shared password required to use the app. Empty = no password (fine for local use;
# set one before exposing the app publicly, especially with OPENAI_API_KEY set).
ACCESS_PASSWORD = os.getenv("EXAMFORGE_PASSWORD", "")

# App-wide cap on AI requests per rolling hour that use the server's resources
# (its OpenAI key or Ollama). Visitors using their own OpenAI key aren't counted.
# 0 = unlimited.
MAX_AI_REQUESTS_PER_HOUR = _int("EXAMFORGE_MAX_AI_REQUESTS_PER_HOUR", 120)

# MongoDB connection string (e.g. an Atlas mongodb+srv://... URI). When set,
# ExamForge has user accounts and saves each user's papers; EXAMFORGE_PASSWORD
# is then not used. Empty = no accounts (the app works as a single shared tool).
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "").strip() or "examforge"

# Per-user cap on AI requests per rolling hour (accounts mode only), on top of
# the app-wide cap. Requests made with the user's own OpenAI key aren't counted.
MAX_AI_REQUESTS_PER_USER_PER_HOUR = _int("EXAMFORGE_MAX_AI_REQUESTS_PER_USER_PER_HOUR", 30)

# How long a sign-in lasts ("stay signed in", survives page reloads), in days.
SESSION_DAYS = _int("EXAMFORGE_SESSION_DAYS", 7, minimum=1)

# Seconds to wait for a single AI request before giving up.
LLM_TIMEOUT_SECONDS = _int("EXAMFORGE_LLM_TIMEOUT", 120, minimum=1)

# Limits for reading an uploaded PDF. Parsing runs in a separate process that is
# stopped after PDF_TIMEOUT seconds or if it needs more than PDF_MEMORY_MB.
PDF_MAX_PAGES = _int("EXAMFORGE_PDF_MAX_PAGES", 50, minimum=1)
PDF_TIMEOUT_SECONDS = _int("EXAMFORGE_PDF_TIMEOUT", 30, minimum=1)
PDF_MEMORY_MB = _int("EXAMFORGE_PDF_MEMORY_MB", 512, minimum=128)

# Limits on the marks table, to keep prompts and costs bounded.
MAX_QUESTIONS = _int("EXAMFORGE_MAX_QUESTIONS", 50, minimum=1)
MAX_MARKS_PER_QUESTION = _int("EXAMFORGE_MAX_MARKS_PER_QUESTION", 100, minimum=1)
