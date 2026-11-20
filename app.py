import html
import logging
import math
import os

import streamlit as st
import pandas as pd

import settings
from security import LoginThrottle, RateLimiter, password_matches
from extractor import extract_text_from_pdf
from syllabus_parser import CHUNK_CHARS, MAX_CHUNKS, is_truncated, parse_syllabus_structure, split_text
from distribution import distribute_questions, validate_distribution
from ai_engine import generate_questions
from pdf_export import generate_pdf
from utils import OLLAMA_BASE_URL, get_available_ollama_models

log = logging.getLogger("examforge")
if not log.handlers:  # app.py re-runs on every interaction; set up once per process
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Server-side OpenAI key (env var or .env). Used as a fallback and never put
# into a widget, because widget values are sent to the browser.
ENV_OPENAI_KEY = settings.OPENAI_API_KEY


@st.cache_resource
def get_ai_limiter() -> RateLimiter:
    """One limiter shared by every session, so it caps the app's total AI usage."""
    if ENV_OPENAI_KEY and not settings.ACCESS_PASSWORD:
        log.warning("OPENAI_API_KEY is set but EXAMFORGE_PASSWORD is not: anyone who can "
                    "reach this app can use your OpenAI key. Set EXAMFORGE_PASSWORD before exposing it.")
    return RateLimiter(settings.MAX_AI_REQUESTS_PER_HOUR)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExamForge - AI Question Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ──────────────────────────────────────────────────────────────
# Colors, fonts and radii come from .streamlit/config.toml (light + dark theme).
# Keep custom CSS color-neutral so it works in both themes.
st.markdown("""
<style>
    /* Tabs: a little larger, bolder and more spaced than Streamlit's default */
    .stTabs [role="tablist"] {
        gap: 1.75rem;
    }
    .stTabs [role="tab"] p {
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Marks label next to each question: keep "[10 Marks]" on one line */
    .marks-label {
        white-space: nowrap;
        text-align: right;
        font-weight: 700;
    }

    /* Sidebar logo shown as a rounded app-icon tile */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 22%;
    }
    .sidebar-brand { text-align: center; margin: 0; }
    .sidebar-tagline { text-align: center; opacity: 0.65; margin-top: 0.25rem; }

    /* Backend status dots */
    .status-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot-green { background: #10b981; }
    .dot-red   { background: #ef4444; }
    .dot-blue  { background: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ── Access password (only when EXAMFORGE_PASSWORD is set) ─────────────────────
def render_login() -> None:
    throttle = st.session_state.setdefault("login_throttle", LoginThrottle())
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.title("ExamForge")
        st.caption("This app is password protected.")
        with st.form("login"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            wait = throttle.seconds_locked()
            if wait:
                st.error(f"Too many attempts. Try again in {math.ceil(wait)} seconds.")
            elif password_matches(password, settings.ACCESS_PASSWORD):
                throttle.record_success()
                st.session_state.authenticated = True
                st.rerun()
            else:
                throttle.record_failure()
                log.warning("Failed sign-in attempt")
                st.error("Incorrect password.")


if settings.ACCESS_PASSWORD and not st.session_state.get("authenticated"):
    render_login()
    st.stop()

# ── Session state defaults ─────────────────────────────────────────────────────
DEFAULTS = {
    "syllabus_text": "",
    "topics": [],
    "generated_questions": [],
    "exam_name": "Midterm Examination",
    "subject_name": "Computer Science",
    "difficulty": "Medium",
    "marks_distribution": pd.DataFrame({"Marks": [5, 10], "Questions": [4, 2]}),
    "use_openai": False,
    "openai_key": "",
    "openai_model": "gpt-4o-mini",
    "ollama_model": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Streamlit discards a keyed widget's value whenever that widget isn't rendered
# in a run (e.g. the OpenAI key field while Ollama is selected). Re-assigning
# the values at the top of every run stops that, so switching backends keeps them.
WIDGET_KEYS = ["use_openai", "openai_key", "openai_model", "ollama_model",
               "exam_name", "subject_name", "difficulty"]
for k in WIDGET_KEYS:
    st.session_state[k] = st.session_state[k]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(APP_DIR, "logo.png")
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(logo_path, width="stretch")
            
    st.markdown("<h2 class='sidebar-brand'>ExamForge</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-tagline'>AI-Powered Question Generator</p>", unsafe_allow_html=True)
    st.divider()

    st.toggle(
        "Use OpenAI API Key",
        key="use_openai",
        help="Turn on to use the OpenAI cloud API instead of the local Ollama model."
    )

    if st.session_state.use_openai:
        st.markdown('<span class="status-dot dot-blue"></span>**OpenAI (Cloud)**', unsafe_allow_html=True)
        st.text_input(
            "OpenAI API Key",
            key="openai_key",
            type="password",
            placeholder="sk-...",
            help="Your key is never stored or sent anywhere except the OpenAI API."
        )
        st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], key="openai_model")
        
        if st.session_state.openai_key:
            st.success("API key entered.")
        elif ENV_OPENAI_KEY:
            st.success("Using the server's API key.")
        else:
            st.warning("Enter your OpenAI API key to continue.")

    else:
        available_models = get_available_ollama_models()
        if available_models:
            st.markdown('<span class="status-dot dot-green"></span>**Ollama — Running**', unsafe_allow_html=True)
            if st.session_state.get("ollama_model") not in available_models:
                st.session_state["ollama_model"] = available_models[0]
                
            st.selectbox(
                "Local model",
                options=available_models,
                key="ollama_model",
                help="Models installed in your local Ollama instance."
            )
            st.caption(f"Using `{st.session_state.ollama_model}` locally.")
        else:
            st.markdown('<span class="status-dot dot-red"></span>**Ollama — Not detected**', unsafe_allow_html=True)
            st.error(f"Ollama is not reachable at `{OLLAMA_BASE_URL}`.\n\nStart it with `ollama serve`.")
            st.session_state["ollama_model"] = ""

    st.divider()
    if settings.ACCESS_PASSWORD and st.button("Sign out", width="stretch"):
        st.session_state.authenticated = False
        st.rerun()

# ── Guard: ensure a model is ready ────────────────────────────────────────────
_use_openai = st.session_state.use_openai
ready = (
    (_use_openai and (st.session_state.get("openai_key") or ENV_OPENAI_KEY))
    or (not _use_openai and st.session_state.get("ollama_model"))
)

# Note: don't st.stop() here. Streamlit discards the state of keyed widgets that
# aren't rendered in a run, so hiding the tabs would wipe the exam settings when
# the user toggles the backend. Instead the UI always renders and only the
# AI buttons are disabled until a backend is ready.

# ── Helper: build shared AI kwargs ────────────────────────────────────────────
def ai_kwargs() -> dict:
    return {
        "use_openai":   st.session_state.use_openai,
        "openai_key":   st.session_state.get("openai_key", "") or ENV_OPENAI_KEY,
        "openai_model": st.session_state.get("openai_model", "gpt-4o-mini"),
        "ollama_model": st.session_state.get("ollama_model", ""),
    }


def reserve_ai_requests(cost: int) -> bool:
    """
    Count `cost` AI requests against the app-wide hourly limit, showing an error
    if it's reached. Visitors using their own OpenAI key aren't limited: they pay.
    """
    if st.session_state.use_openai and st.session_state.get("openai_key"):
        return True
    allowed, retry_after = get_ai_limiter().try_acquire(cost)
    if not allowed:
        log.warning("AI usage limit reached (%s requests/hour)", settings.MAX_AI_REQUESTS_PER_HOUR)
        st.error(
            f"This app has reached its limit of {settings.MAX_AI_REQUESTS_PER_HOUR} AI requests per hour. "
            f"Try again in about {max(1, math.ceil(retry_after / 60))} minute(s), or use your own OpenAI key."
        )
    return allowed

# ── Main content ───────────────────────────────────────────────────────────────
st.title("ExamForge")
st.markdown("#### Generate professional exam papers in seconds")

if not ready:
    st.info("Please configure your AI backend in the sidebar to analyze syllabi and generate questions.")
elif _use_openai:
    st.caption(f"Backend: OpenAI ({st.session_state.get('openai_model')})")
else:
    st.caption(f"Backend: Local Ollama ({st.session_state.get('ollama_model')})")

tab1, tab2, tab3, tab4 = st.tabs(["1. Upload", "2. Configuration", "3. Generated Paper", "4. Export"])

with tab1:
    st.header("Upload Syllabus PDF")
    st.markdown("Upload your syllabus to extract and analyze its topic structure automatically.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}**")

        if st.button("Extract & Analyze Syllabus", type="primary", disabled=not ready):
            read_error = None
            with st.spinner("Extracting text from PDF..."):
                try:
                    extracted_text = extract_text_from_pdf(uploaded_file)
                except ValueError as e:
                    extracted_text, read_error = "", e
                st.session_state.syllabus_text = extracted_text

            if read_error:
                st.error(f"{read_error}. Is this a valid PDF file?")
            elif not extracted_text.strip():
                st.error("No text could be extracted. The PDF may be scanned or image-based.")
            else:
                if is_truncated(extracted_text):
                    st.warning(
                        f"This syllabus is long; only the first {MAX_CHUNKS * CHUNK_CHARS:,} "
                        f"of {len(extracted_text):,} characters will be analyzed."
                    )
                n_parts = min(len(split_text(extracted_text)), MAX_CHUNKS)
                backend_label = f"OpenAI ({st.session_state.get('openai_model')})" if _use_openai else f"Ollama ({st.session_state.get('ollama_model')})"
                if reserve_ai_requests(n_parts):
                    with st.spinner(f"Analyzing syllabus ({n_parts} part(s)) with {backend_label} — please wait..."):
                        try:
                            topics = parse_syllabus_structure(extracted_text, **ai_kwargs())
                            st.session_state.topics = topics
                            # Questions from a previous syllabus no longer apply.
                            st.session_state.generated_questions = []
                            st.success(f"Done! Discovered **{len(topics)}** topic(s).")
                        except Exception as e:
                            log.exception("Syllabus analysis failed")
                            st.error(f"AI analysis failed: {e}")

    if st.session_state.topics:
        with st.expander("View Discovered Topics", expanded=True):
            for i, t in enumerate(st.session_state.topics):
                st.markdown(f"**{i + 1}. {t.get('topic_name', 'Unknown')}**")
                st.caption(t.get("description", ""))

with tab2:
    st.header("Exam Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Exam Name", key="exam_name")
        st.text_input("Subject Name", key="subject_name")
        st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty")

    with col2:
        st.markdown("**Question Distribution**")
        st.caption("Add rows to define how many questions of each mark value.")
        edited_df = st.data_editor(
            st.session_state.marks_distribution,
            key="marks_editor",
            num_rows="dynamic",
            width="stretch"
        )
        if not edited_df.equals(st.session_state.marks_distribution):
            st.session_state.marks_distribution = edited_df

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Questions", type="primary", disabled=not ready):
        if not st.session_state.topics:
            st.error("Please upload and analyze a syllabus first (Upload tab).")
        else:
            dist_list = [{"marks": int(row["Marks"]), "count": int(row["Questions"])} for _, row in st.session_state.marks_distribution.iterrows() if pd.notna(row["Marks"]) and pd.notna(row["Questions"])]
            errors = validate_distribution(dist_list, settings.MAX_QUESTIONS, settings.MAX_MARKS_PER_QUESTION)

            if errors:
                for err in errors:
                    st.error(err)
            elif reserve_ai_requests(1):
                question_plan = distribute_questions(dist_list, st.session_state.topics)
                backend_label = f"OpenAI" if _use_openai else f"Ollama"
                with st.spinner(f"Generating {len(question_plan)} question(s) with {backend_label}..."):
                    try:
                        questions = generate_questions(question_plan, difficulty=st.session_state.difficulty, **ai_kwargs())
                        st.session_state.generated_questions = questions
                        st.success(f"Generated **{len(questions)}** question(s). Check the 'Generated Paper' tab.")
                    except Exception as e:
                        log.exception("Question generation failed")
                        st.error(f"Question generation failed: {e}")

with tab3:
    st.header("Generated Question Paper")

    if st.session_state.generated_questions:
        total_marks = sum(q.get("marks", 0) for q in st.session_state.generated_questions)
        
        st.markdown(f"### {st.session_state.exam_name}")
        st.markdown(f"**Subject:** {st.session_state.subject_name}")
        st.markdown(f"**Total Marks:** {total_marks} | **Difficulty:** {st.session_state.difficulty}")
        st.divider()

        for idx, q in enumerate(st.session_state.generated_questions, 1):
            col_q, col_m = st.columns([5, 1])
            with col_q:
                st.markdown(f"**Q{idx}.** {q.get('question_text', '')}")
                st.caption(f"Topic: {q.get('topic_name', 'Unknown')}")
            with col_m:
                marks = html.escape(str(q.get('marks', 0)))
                st.markdown(f"<div class='marks-label'>[{marks} Marks]</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:8px 0;opacity:0.15;'>", unsafe_allow_html=True)
    else:
        st.info("No questions generated yet. Go to the 'Configuration' tab to generate.")

with tab4:
    st.header("Export to PDF")

    if st.session_state.generated_questions:
        st.markdown("Your question paper is ready to be exported as a PDF.")
        try:
            pdf_buffer = generate_pdf(
                st.session_state.exam_name,
                st.session_state.subject_name,
                st.session_state.generated_questions
            )
        except Exception as e:
            st.error(f"Could not build the PDF: {e}")
        else:
            st.download_button(
                label="Download Question Paper PDF",
                data=pdf_buffer,
                file_name="QuestionPaper.pdf",
                mime="application/pdf",
                type="primary"
            )
    else:
        st.info("Generate questions first, then export here.")
