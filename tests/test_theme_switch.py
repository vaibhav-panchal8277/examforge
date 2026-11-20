"""ExamForge's System / Light / Dark switch (theme_switch.py) and the setup it depends on."""
import os
import tomllib

import theme_switch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_toolbar_keeps_streamlit_menu_available():
    # The switch operates Streamlit's (hidden) theme menu. "minimal" removes that
    # menu entirely, which would silently break the switch.
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as f:
        config = tomllib.load(f)
    assert config["client"]["toolbarMode"] == "viewer"
    assert "light" in config["theme"] and "dark" in config["theme"]


def test_offers_system_light_dark_as_accessible_radios():
    html = theme_switch._HTML
    assert 'role="radiogroup"' in html and 'aria-label="Theme"' in html
    for choice in ("System", "Light", "Dark"):
        assert f'data-choice="{choice}"' in html
    assert html.count('role="radio"') == 3


def test_relies_only_on_streamlits_named_menu_hooks():
    js = theme_switch._JS
    assert '[data-testid="stMainMenuButton"]' in js
    assert "stMainMenuItem-theme-${choice}" in js
    assert "stActiveTheme-" in js  # Streamlit's saved choice, read to show the current theme


def test_hides_streamlits_menu():
    css = theme_switch.MENU_HIDE_CSS
    assert '[data-testid="stMainMenu"]' in css and "display: none" in css


def test_falls_back_to_streamlits_menu_if_hooks_change():
    # If the menu can't be found, the switch disables itself and un-hides Streamlit's menu.
    js = theme_switch._JS
    assert "markUnavailable" in js and "s.disabled = true" in js
