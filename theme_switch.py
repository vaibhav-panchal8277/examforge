"""
ExamForge's own System / Light / Dark theme switch.

Streamlit has no Python API for changing the theme, and reloading the page to
apply one would start a new session (signing the user out). So this switch
drives Streamlit's own theme engine from the browser: it operates Streamlit's
built-in theme menu, which ExamForge keeps hidden. The result is the same as
using that menu: the theme changes instantly, the session is kept, and the
choice is remembered in the browser.

It relies on Streamlit's menu markup (data-testid="stMainMenuButton" and
"stMainMenuItem-theme-<System|Light|Dark>", present in Streamlit 1.67). If a
future Streamlit version changes those, the switch disables itself, shows a
tooltip, and the hidden menu stops being hidden (see MENU_HIDE_CSS), so users
can still change the theme there.
"""
import streamlit as st

# Hide Streamlit's ⋮ menu (theme switch, Print, Record screen). The theme part
# is reached through ExamForge's own switch instead. Also pins the switch,
# rendered in the "theme_corner" container, to the top-left corner of the page.
MENU_HIDE_CSS = """
<style>
    [data-testid="stMainMenu"], [data-testid="stMainMenuPopover"] { display: none !important; }
    .st-key-theme_corner {
        position: fixed;
        top: 0.7rem;
        left: 1rem;
        z-index: 1000002;  /* above Streamlit's header */
        width: auto !important;
    }
    /* With the sidebar closed, Streamlit's "open sidebar" button sits in the
       corner: keep the switch to its right so it never covers it. */
    body:has([data-testid="stExpandSidebarButton"]) .st-key-theme_corner { left: 3.4rem; }
</style>
"""

_ICONS = {  # Lucide icons (MIT licence), drawn in the current text colour
    "System": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/>'
              '<line x1="12" x2="12" y1="17" y2="21"/>',
    "Light": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/>'
             '<path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/>'
             '<path d="m19.07 4.93-1.41 1.41"/>',
    "Dark": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
}

_HTML = '<div class="ef-wrap"><div class="ef-theme" role="radiogroup" aria-label="Theme">' + "".join(
    f'<button type="button" role="radio" data-choice="{name}" aria-checked="false" title="{name} theme">'
    f'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg><span>{name}</span></button>'
    for name, path in _ICONS.items()
) + "</div></div>"

_CSS = """
/* Alignment is set by the page through the inherited --ef-align property. */
.ef-wrap { display: flex; justify-content: var(--ef-align, flex-start); }
.ef-theme {
    display: inline-flex;
    gap: 2px;
    padding: 3px;
    border-radius: 999px;
    border: 1px solid var(--st-border-color, rgba(128, 128, 128, 0.25));
    background: var(--st-secondary-background-color, rgba(128, 128, 128, 0.08));
    font-family: inherit;
}
.ef-theme button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border: 0;
    border-radius: 999px;
    background: transparent;
    color: var(--st-text-color, inherit);
    opacity: 0.7;
    font: inherit;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
}
.ef-theme button:hover { opacity: 1; }
.ef-theme button:focus-visible { outline: 2px solid var(--st-primary-color, #6366f1); outline-offset: 1px; }
.ef-theme button[aria-checked="true"] {
    opacity: 1;
    background: var(--st-background-color, #ffffff);
    color: var(--st-primary-color, #4f46e5);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.18);
}
.ef-theme.unavailable button { cursor: not-allowed; opacity: 0.4; }
/* Compact: icons only (each still has a tooltip and an accessible name). */
.ef-theme.compact button { padding: 6px 9px; }
.ef-theme.compact button span {
    position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;
}
"""

_JS = """
const PREFIX = `stActiveTheme-${window.location.pathname}-v`;   // Streamlit's own saved choice
const CHOICES = ["System", "Light", "Dark"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function savedChoice() {
    // Streamlit stores e.g. "Dark" under stActiveTheme-/-v2; take the newest version present.
    const keys = Object.keys(localStorage).filter((k) => k.startsWith(PREFIX))
        .sort((a, b) => Number(b.slice(PREFIX.length)) - Number(a.slice(PREFIX.length)));
    try {
        const value = keys.length ? JSON.parse(localStorage.getItem(keys[0])) : "System";
        return CHOICES.includes(value) ? value : "System";
    } catch (e) {
        return "System";
    }
}

async function applyTheme(choice) {
    // Open Streamlit's (hidden) menu, pick the theme, close the menu.
    const menuButton = document.querySelector('[data-testid="stMainMenuButton"]');
    if (!menuButton) return false;
    menuButton.click();
    let item = null;
    for (let i = 0; i < 40 && !item; i++) {
        await sleep(25);
        item = document.querySelector(`[data-testid="stMainMenuItem-theme-${choice}"]`);
    }
    if (!item) {
        if (document.querySelector('[data-testid="stMainMenuPopover"]')) menuButton.click();
        return false;
    }
    item.click();
    await sleep(30);
    if (document.querySelector('[data-testid="stMainMenuPopover"]')) menuButton.click();
    return true;
}

function markUnavailable(root) {
    root.classList.add("unavailable");
    root.title = "Theme switching isn't available in this Streamlit version.";
    // Let users fall back to Streamlit's own menu.
    document.querySelectorAll("style").forEach((s) => {
        if (s.textContent.includes('[data-testid="stMainMenu"]')) s.disabled = true;
    });
}

export default function (component) {
    const root = component.parentElement.querySelector(".ef-theme");
    root.classList.toggle("compact", Boolean(component.data && component.data.compact));
    const buttons = [...root.querySelectorAll("button[data-choice]")];
    const show = (choice) => buttons.forEach((b) =>
        b.setAttribute("aria-checked", String(b.dataset.choice === choice)));
    show(savedChoice());

    buttons.forEach((b) => {
        b.onclick = async () => {
            if (root.classList.contains("unavailable")) return;
            const choice = b.dataset.choice;
            show(choice);
            if (await applyTheme(choice)) {
                window.dispatchEvent(new CustomEvent("examforge-theme", { detail: choice }));
            } else {
                console.warn("ExamForge: Streamlit's theme menu wasn't found; theme switch disabled.");
                markUnavailable(root);
                show(savedChoice());
            }
        };
    });
    // Keep every switch on the page (sidebar, sign-in page) in step. Attached
    // once per element, since Streamlit may call this again on each redraw.
    if (!root.dataset.listening) {
        root.dataset.listening = "1";
        window.addEventListener("examforge-theme", (e) => {
            if (root.isConnected) show(e.detail);
        });
    }
}
"""

_component = st.components.v2.component("examforge_theme_switch", html=_HTML, css=_CSS, js=_JS)


def theme_switch(key: str, compact: bool = False) -> None:
    """Render ExamForge's System / Light / Dark switch (works entirely in the browser).
    `compact` shows icons only, with each option's name as a tooltip."""
    _component(key=key, data={"compact": compact})
