"""
The "stay signed in" cookie.

Streamlit can read the browser's cookies (st.context.cookies) but has no way to
set them, so this tiny, invisible component sets or clears the cookie from the
browser. The cookie only holds an opaque session token; what it grants is
decided on the server (accounts.Store.user_for_session or
security.verify_password_session).
"""
import streamlit as st

COOKIE_NAME = "examforge_session"

# Restricted cookie: same-site requests only (so other websites can't send it),
# and Secure over HTTPS. It can't be HttpOnly because it's set from JavaScript.
_JS = """
export default function (component) {
    const { data } = component;
    if (!data || !data.name) return;
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    if (data.action === "set") {
        document.cookie = `${data.name}=${encodeURIComponent(data.token)}; Max-Age=${data.maxAge}; Path=/; SameSite=Strict${secure}`;
    } else if (data.action === "clear") {
        document.cookie = `${data.name}=; Max-Age=0; Path=/; SameSite=Strict${secure}`;
    }
}
"""

_component = st.components.v2.component("examforge_session_cookie", js=_JS)


def read_token() -> str:
    """The session token the browser sent with this page load ("" if none)."""
    try:
        token = st.context.cookies.get(COOKIE_NAME, "")
    except Exception:  # no browser context (e.g. headless tests)
        return ""
    return token if isinstance(token, str) else ""


def set_cookie(token: str, days: int) -> None:
    _component(key="session_cookie", data={"name": COOKIE_NAME, "action": "set",
                                           "token": token, "maxAge": days * 86400})


def clear_cookie() -> None:
    _component(key="session_cookie", data={"name": COOKIE_NAME, "action": "clear"})
