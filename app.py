import base64
import functools
import html
import logging
import math
import os
import re

import streamlit as st
import pandas as pd

import accounts
import settings
import session_cookie
from security import (KeyedRegistry, LoginThrottle, RateLimiter, password_matches,
                      sign_password_session, verify_password_session)
from theme_switch import MENU_HIDE_CSS, theme_switch
from extractor import PDFError, extract_text_safely
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
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo-192.png")  # 14 KB; logo.png is the 1024 px original


@functools.lru_cache(maxsize=None)
def logo_img(size: int, css_class: str) -> str:
    """
    The logo as an inline <img> with a fixed size. It arrives with the page
    itself: no separate request, no server-side image processing on every run
    (as st.image does), and no layout jump while it loads.
    """
    try:
        with open(LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return ""
    return (f"<img class='{css_class}' src='data:image/png;base64,{data}' "
            f"width='{size}' height='{size}' alt='ExamForge logo'>")

# Server-side OpenAI key (env var or .env). Used as a fallback and never put
# into a widget, because widget values are sent to the browser.
ENV_OPENAI_KEY = settings.OPENAI_API_KEY

# With a MongoDB connection, ExamForge has user accounts and saves each user's papers.
ACCOUNTS_ENABLED = bool(settings.MONGODB_URI)


@st.cache_resource
def get_ai_limiter() -> RateLimiter:
    """One limiter shared by every session, so it caps the app's total AI usage."""
    if ENV_OPENAI_KEY and ACCOUNTS_ENABLED:
        log.info("Accounts are open for sign-up and OPENAI_API_KEY is set: signed-in users can use "
                 "your OpenAI key, up to %s requests/hour each and %s/hour in total.",
                 settings.MAX_AI_REQUESTS_PER_USER_PER_HOUR, settings.MAX_AI_REQUESTS_PER_HOUR)
    elif ENV_OPENAI_KEY and not settings.ACCESS_PASSWORD:
        log.warning("OPENAI_API_KEY is set but EXAMFORGE_PASSWORD is not: anyone who can "
                    "reach this app can use your OpenAI key. Set EXAMFORGE_PASSWORD before exposing it.")
    return RateLimiter(settings.MAX_AI_REQUESTS_PER_HOUR)


@st.cache_resource(show_spinner=False)
def get_store() -> accounts.Store:
    """One MongoDB connection pool shared by every session."""
    return accounts.connect(settings.MONGODB_URI, settings.MONGODB_DB)


@st.cache_resource
def get_user_limiters() -> KeyedRegistry:
    """Per-user hourly AI limits (accounts mode)."""
    return KeyedRegistry(lambda: RateLimiter(settings.MAX_AI_REQUESTS_PER_USER_PER_HOUR))


@st.cache_resource
def get_login_throttles() -> KeyedRegistry:
    """Per-email sign-in lockouts, so a new browser session doesn't reset them."""
    return KeyedRegistry(LoginThrottle)

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

    /* Sidebar logo shown as a rounded app-icon tile (fixed size, so nothing shifts) */
    .sidebar-logo { text-align: center; margin: 0.5rem 0 0.75rem; }
    .sidebar-logo img { border-radius: 22%; }
    .sidebar-brand { text-align: center; margin: 0; }
    .sidebar-tagline { text-align: center; opacity: 0.65; margin-top: 0.25rem; }
    .signed-in { font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.5rem; overflow-wrap: anywhere; }

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

# ── Sign-in screens (accounts, or the shared password) ────────────────────────
# Shown only on the sign-in screens: a branded panel beside a form card.
# The panel uses a fixed indigo gradient with white text, so it reads the same
# in light and dark mode; the card follows the theme.
AUTH_CSS = """
<style>
    /* Narrower, centred page only while the sign-in panel is on screen */
    .block-container:has(.st-key-auth_hero) { max-width: 1120px; padding-top: 3.5rem; }
    .st-key-auth_hero {
        position: relative;
        overflow: hidden;
        padding: 2.75rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(145deg, #4338ca 0%, #5b21b6 60%, #7c3aed 100%);
        box-shadow: 0 24px 48px -24px rgba(67, 56, 202, 0.65);
    }
    .st-key-auth_hero::after {  /* soft decorative glow */
        content: "";
        position: absolute;
        top: -90px; right: -90px;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.20), transparent 70%);
        pointer-events: none;
    }
    .st-key-auth_hero, .st-key-auth_hero * { color: #ffffff !important; }
    .auth-logo {
        display: block;
        border-radius: 22%;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.30);
    }
    .auth-title { font-size: 2.5rem; font-weight: 700; line-height: 1.1; margin: 0.5rem 0 0.5rem; }
    .auth-tagline { font-size: 1.1rem; line-height: 1.5; opacity: 0.92; margin-bottom: 1.75rem; }
    .auth-feature { display: flex; gap: 0.85rem; align-items: flex-start; margin-bottom: 1.1rem; }
    .auth-feature:last-child { margin-bottom: 0; }
    .auth-feature-icon {
        flex: 0 0 2.4rem; height: 2.4rem;
        display: flex; align-items: center; justify-content: center;
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .auth-feature b { display: block; font-size: 0.98rem; }
    .auth-feature span { display: block; font-size: 0.88rem; opacity: 0.82; margin-top: 0.1rem; }
    .st-key-auth_card { padding: 1.5rem 1.5rem 1rem; }
    .st-key-auth_card .auth-card-title { font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
    .st-key-auth_card .auth-card-subtitle { font-size: 0.95rem; opacity: 0.7; margin: 0.35rem 0 0.5rem; }
    .st-key-auth_card .auth-footnote { font-size: 0.78rem; opacity: 0.6; text-align: center; margin: 0.25rem 0 0.5rem; }
    @media (max-width: 640px) {
        .block-container:has(.st-key-auth_hero) { padding-top: 1.25rem; }
        .st-key-auth_hero { padding: 1.75rem 1.5rem; }
        .auth-title { font-size: 2rem; }
        .auth-tagline { margin-bottom: 0; }
        .auth-features { display: none; }  /* phones: brand only, then the form */
    }
</style>
"""

# Sent at the top of every page, before any content, so the sign-in layout is
# styled from its first frame (styles that arrive later make the page jump).
# Every rule is scoped to the sign-in elements, so the main app is unaffected.
st.markdown(AUTH_CSS + MENU_HIDE_CSS, unsafe_allow_html=True)

# Line icons (Lucide, MIT licence) drawn in the current text colour.
_ICON_PATHS = {
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
            '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/>'
               '<line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/>'
               '<line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/>'
               '<line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
               '<line x1="16" x2="16" y1="18" y2="22"/>',
    "bookmark": '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/>'
                '<line x1="12" x2="12" y1="15" y2="3"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
}


def _feature(icon: str, title: str, text: str) -> str:
    svg = ('<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[icon]}</svg>')
    return (f"<div class='auth-feature'><div class='auth-feature-icon'>{svg}</div>"
            f"<div><b>{title}</b><span>{text}</span></div></div>")


def auth_layout(features: list, card_min_height: int):
    """
    Draw the brand panel and return the card container the sign-in form goes in.

    `card_min_height` reserves the card's final height (px, desktop), so the card
    doesn't start small and grow as its fields arrive, which would shift it.
    """
    st.markdown(f"<style>@media (min-width: 641px) {{ .st-key-auth_card "
                f"{{ min-height: {card_min_height}px; }} }}</style>", unsafe_allow_html=True)
    hero_col, form_col = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with hero_col:
        with st.container(key="auth_hero"):
            # One element (logo, text and features together), so the panel is
            # drawn complete in a single step.
            st.markdown(
                logo_img(68, "auth-logo") +
                "<div class='auth-title'>ExamForge</div>"
                "<div class='auth-tagline'>Turn any syllabus into a balanced, print-ready exam paper in minutes.</div>"
                "<div class='auth-features'>" + "".join(_feature(*f) for f in features) + "</div>",
                unsafe_allow_html=True,
            )
    return form_col.container(border=True, key="auth_card")


# Final card heights at desktop width (Sign in tab / password form), measured in
# the browser. Update these if the card's content changes.
ACCOUNT_CARD_HEIGHT = 384
PASSWORD_CARD_HEIGHT = 276


def _card_heading(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='auth-card-title'>{title}</div><div class='auth-card-subtitle'>{subtitle}</div>",
                unsafe_allow_html=True)


# ── Access password (only when EXAMFORGE_PASSWORD is set) ─────────────────────
def render_login() -> None:
    throttle = st.session_state.setdefault("login_throttle", LoginThrottle())
    card = auth_layout([
        ("file", "Upload any syllabus", "Topics are found automatically from your PDF."),
        ("sliders", "Set marks &amp; difficulty", "Questions follow your exact marks distribution."),
        ("download", "Print-ready PDFs", "Download a clean question paper in one click."),
        ("lock", "Private access", "Only people with the password can use this app."),
    ], card_min_height=PASSWORD_CARD_HEIGHT)
    with card:
        _card_heading("Welcome", "This app is password protected. Enter the password to continue.")
        with st.form("login", border=False):
            password = st.text_input("Password", type="password", placeholder="Enter the app password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            wait = throttle.seconds_locked()
            if wait:
                st.error(f"Too many attempts. Try again in {math.ceil(wait)} seconds.")
            elif password_matches(password, settings.ACCESS_PASSWORD):
                throttle.record_success()
                st.session_state.authenticated = True
                _remember_session(sign_password_session(settings.ACCESS_PASSWORD, settings.SESSION_DAYS))
                st.rerun()
            else:
                throttle.record_failure()
                log.warning("Failed sign-in attempt")
                st.error("Incorrect password.")
        st.markdown("<div class='auth-footnote'>Don't have the password? Ask the person who runs this app.</div>",
                    unsafe_allow_html=True)


# ── "Stay signed in" sessions ─────────────────────────────────────────────────
# Streamlit forgets everything when the page is reloaded, so a sign-in is also
# kept in a browser cookie holding a session token. On reload the token is
# checked on the server and the user is signed straight back in.
def _remember_session(token: str) -> None:
    st.session_state.session_token = token
    st.session_state.signed_out = False
    st.session_state.clear_session_cookie = False


def start_session(user: dict, store: accounts.Store) -> None:
    st.session_state.user = user
    _remember_session(store.create_session(user["id"], settings.SESSION_DAYS))
    st.rerun()


def sign_out() -> None:
    """End the session everywhere, and clear everything so the next person on this browser sees nothing."""
    token = st.session_state.get("session_token") or session_cookie.read_token()
    if ACCOUNTS_ENABLED and token:
        try:
            get_store().delete_session(token)
        except Exception:
            log.exception("Ending a session failed")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # This page still carries the old cookie until it reloads, so ignore it and clear it.
    st.session_state.signed_out = True
    st.session_state.clear_session_cookie = True


def restore_session(store) -> None:
    """Sign the user back in from the browser's session cookie, if it's valid."""
    if st.session_state.get("signed_out") or st.session_state.get("user") or st.session_state.get("authenticated"):
        return
    token = session_cookie.read_token()
    if not token:
        return
    if store is not None:
        user = store.user_for_session(token)
        if user:
            st.session_state.user = user
    elif settings.ACCESS_PASSWORD and verify_password_session(token, settings.ACCESS_PASSWORD):
        st.session_state.authenticated = True
    if st.session_state.get("user") or st.session_state.get("authenticated"):
        _remember_session(token)
        log.info("Signed in from a saved session")
    else:  # expired, signed out elsewhere, or tampered with
        st.session_state.signed_out = True
        st.session_state.clear_session_cookie = True


def sync_session_cookie() -> None:
    """Keep the browser's cookie in step with the sign-in state (invisible)."""
    if st.session_state.get("session_token"):
        session_cookie.set_cookie(st.session_state.session_token, settings.SESSION_DAYS)
    elif st.session_state.get("clear_session_cookie"):
        session_cookie.clear_cookie()


# ── User accounts (only when MONGODB_URI is set) ──────────────────────────────


def render_account_screen(store: accounts.Store) -> None:
    card = auth_layout([
        ("file", "Upload any syllabus", "Topics are found automatically from your PDF."),
        ("sliders", "Set marks &amp; difficulty", "Questions follow your exact marks distribution."),
        ("bookmark", "Saved to your account", "Reopen, re-download or delete papers any time."),
        ("lock", "Private by design", "Your papers are visible only to you."),
    ], card_min_height=ACCOUNT_CARD_HEIGHT)
    with card:
        _card_heading("Welcome", "Sign in, or create a free account to get started.")
        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("sign_in", border=False):
                email = st.text_input("Email", key="signin_email", placeholder="you@school.edu")
                password = st.text_input("Password", type="password", key="signin_password",
                                         placeholder="Enter your password")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            if submitted:
                throttle = get_login_throttles().get(accounts.normalize_email(email))
                wait = throttle.seconds_locked()
                if wait:
                    st.error(f"Too many attempts. Try again in {math.ceil(wait)} seconds.")
                else:
                    user = store.authenticate(email, password)
                    if user:
                        throttle.record_success()
                        start_session(user, store)
                    else:
                        throttle.record_failure()
                        log.warning("Failed sign-in attempt")
                        st.error("Incorrect email or password.")

        with sign_up_tab:
            with st.form("sign_up", border=False):
                new_email = st.text_input("Email", key="signup_email", placeholder="you@school.edu")
                new_password = st.text_input(
                    "Password", type="password", key="signup_password",
                    placeholder=f"At least {accounts.MIN_PASSWORD_LENGTH} characters")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm",
                                        placeholder="Re-enter your password")
                created = st.form_submit_button("Create account", type="primary", width="stretch")
            if created:
                if new_password != confirm:
                    st.error("Passwords don't match.")
                else:
                    try:
                        user = store.create_user(new_email, new_password)
                    except accounts.AccountError as e:
                        st.error(str(e))
                    else:
                        log.info("New account created")
                        start_session(user, store)

        st.markdown("<div class='auth-footnote'>Passwords are stored securely hashed, never in plain text.</div>",
                    unsafe_allow_html=True)

# ── Theme switch, top-left corner of every page (sign-in screens and the app) ──
with st.container(key="theme_corner"):
    theme_switch(key="theme_corner_switch", compact=True)

# ── Sign-in gate ──────────────────────────────────────────────────────────────
store = None
if ACCOUNTS_ENABLED:
    try:
        store = get_store()
    except Exception:
        log.exception("Could not connect to MongoDB")
        st.error("ExamForge can't reach its database right now. Please try again in a moment.")
        st.stop()
if ACCOUNTS_ENABLED or settings.ACCESS_PASSWORD:
    restore_session(store)
    sync_session_cookie()

if ACCOUNTS_ENABLED and not st.session_state.get("user"):
    render_account_screen(store)
    st.stop()
elif not ACCOUNTS_ENABLED and settings.ACCESS_PASSWORD and not st.session_state.get("authenticated"):
    render_login()
    st.stop()

current_user = st.session_state.get("user")  # {"id", "email"} in accounts mode, else None

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
    "current_paper_id": None,  # saved paper shown in Generated Paper (accounts mode)
    "flash": None,             # one-off message shown on the next run
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
    st.markdown(f"<div class='sidebar-logo'>{logo_img(96, '')}</div>", unsafe_allow_html=True)
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
    if current_user:
        # Raw HTML so Markdown doesn't turn the email address into a mailto link.
        st.markdown(
            f"<div class='signed-in'>Signed in as<br><b>{html.escape(current_user['email'])}</b></div>",
            unsafe_allow_html=True)
    if current_user or settings.ACCESS_PASSWORD:
        st.button("Sign out", width="stretch", on_click=sign_out)

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
    Count `cost` AI requests against the hourly limits (per user in accounts mode,
    then app-wide), showing an error if one is reached. Visitors using their own
    OpenAI key aren't limited: they pay.
    """
    if st.session_state.use_openai and st.session_state.get("openai_key"):
        return True
    if current_user:
        allowed, retry_after = get_user_limiters().get(current_user["id"]).try_acquire(cost)
        if not allowed:
            log.warning("Per-user AI limit reached (%s requests/hour)", settings.MAX_AI_REQUESTS_PER_USER_PER_HOUR)
            st.error(
                f"You've reached your limit of {settings.MAX_AI_REQUESTS_PER_USER_PER_HOUR} AI requests per hour. "
                f"Try again in about {max(1, math.ceil(retry_after / 60))} minute(s), or use your own OpenAI key."
            )
            return False
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

if st.session_state.flash:
    st.toast(st.session_state.flash)
    st.session_state.flash = None

tab_names = ["1. Upload", "2. Configuration", "3. Generated Paper", "4. Export"]
if current_user:
    tab_names.append("My Papers")
tab1, tab2, tab3, tab4, *extra_tabs = st.tabs(tab_names)

with tab1:
    st.header("Upload Syllabus PDF")
    st.markdown("Upload your syllabus to extract and analyze its topic structure automatically.")
    if _use_openai:
        st.caption("Privacy: the syllabus text is sent to OpenAI for analysis. "
                   "With Ollama it stays on this server. Uploaded files aren't stored.")
    else:
        st.caption("Privacy: the syllabus is analyzed on this server with Ollama. Uploaded files aren't stored.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}**")

        if st.button("Extract & Analyze Syllabus", type="primary", disabled=not ready):
            read_error = None
            with st.spinner("Extracting text from PDF..."):
                try:
                    # Parsed in a separate, time- and memory-limited process (see extractor.py).
                    extracted_text = extract_text_safely(
                        uploaded_file.getvalue(),
                        max_pages=settings.PDF_MAX_PAGES,
                        # One chunk more than is analyzed, so long syllabi still get the warning below.
                        max_chars=(MAX_CHUNKS + 1) * CHUNK_CHARS,
                        timeout_seconds=settings.PDF_TIMEOUT_SECONDS,
                        memory_mb=settings.PDF_MEMORY_MB,
                    )
                except PDFError as e:
                    extracted_text, read_error = "", e
                    log.warning("Rejected uploaded PDF: %s", e)
                st.session_state.syllabus_text = extracted_text

            if read_error:
                st.error(str(read_error))
            elif not extracted_text.strip():
                st.error("No text could be extracted. The PDF may be scanned or image-based.")
            else:
                if is_truncated(extracted_text):
                    st.warning(
                        f"This syllabus is long; only the first {MAX_CHUNKS * CHUNK_CHARS:,} "
                        "characters will be analyzed."
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
                    except Exception as e:
                        log.exception("Question generation failed")
                        st.error(f"Question generation failed: {e}")
                    else:
                        st.session_state.generated_questions = questions
                        st.session_state.current_paper_id = None
                        saved_note = ""
                        if current_user:
                            try:
                                st.session_state.current_paper_id = store.save_paper(
                                    current_user["id"], st.session_state.exam_name,
                                    st.session_state.subject_name, st.session_state.difficulty, questions)
                                saved_note = " and saved it to **My Papers**"
                            except Exception:
                                log.exception("Saving a paper failed")
                                st.warning("The paper couldn't be saved to your account right now. "
                                           "You can still view and export it.")
                        st.success(f"Generated **{len(questions)}** question(s){saved_note}. "
                                   "Check the 'Generated Paper' tab.")

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


# ── My Papers (accounts mode) ─────────────────────────────────────────────────
def pdf_file_name(exam_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", exam_name).strip("_")[:60]
    return f"{name or 'QuestionPaper'}.pdf"


def open_paper(paper: dict) -> None:
    # Runs as a button callback, before widgets are drawn, so it may set widget values.
    st.session_state.generated_questions = paper["questions"]
    st.session_state.exam_name = paper["exam_name"]
    st.session_state.subject_name = paper["subject_name"]
    st.session_state.difficulty = paper["difficulty"]
    st.session_state.current_paper_id = paper["id"]
    st.session_state.flash = f"Opened “{paper['exam_name']}”. See the Generated Paper tab."


def delete_paper(paper_id: str) -> None:
    try:
        deleted = store.delete_paper(current_user["id"], paper_id)
    except Exception:
        log.exception("Deleting a paper failed")
        deleted = False
    st.session_state.flash = "Paper deleted." if deleted else "The paper couldn't be deleted. Please try again."


if current_user:
    with extra_tabs[0]:
        st.header("My Papers")
        try:
            papers = store.list_papers(current_user["id"])
        except Exception:
            log.exception("Loading papers failed")
            st.error("Your papers couldn't be loaded right now. Please try again in a moment.")
            papers = None

        if papers == []:
            st.info("Papers you generate are saved here automatically.")
        elif papers:
            st.caption(f"Your {len(papers)} most recent paper(s), newest first.")
        for paper in papers or []:
            with st.container(border=True):
                st.markdown(f"**{html.escape(paper['exam_name'])}**")
                created = paper["created_at"].strftime("%d %b %Y, %H:%M UTC")
                st.caption(
                    f"{paper['subject_name']} · {paper['total_marks']} marks · "
                    f"{len(paper['questions'])} questions · {paper['difficulty']} · {created}"
                )
                # A row that sizes buttons to their labels and wraps on narrow screens.
                with st.container(horizontal=True):
                    st.button("Open", key=f"open_{paper['id']}", icon=":material/open_in_new:",
                              on_click=open_paper, args=(paper,))
                    try:
                        st.download_button(
                            "Download PDF", key=f"pdf_{paper['id']}", icon=":material/download:",
                            data=generate_pdf(paper["exam_name"], paper["subject_name"], paper["questions"]),
                            file_name=pdf_file_name(paper["exam_name"]), mime="application/pdf")
                    except Exception:
                        log.exception("Building a saved paper's PDF failed")
                        st.caption("PDF unavailable")
                    with st.popover("Delete", icon=":material/delete:"):
                        st.markdown("Delete this paper permanently? This can't be undone.")
                        st.button("Delete paper", key=f"delete_{paper['id']}", type="primary",
                                  on_click=delete_paper, args=(paper["id"],))
