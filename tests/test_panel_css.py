"""Nothing written for a phone may reach the panel.

The admin page is one design served to two very different clients: the Pi's 800x480 kiosk, and
whatever is on the LAN. Everything the second one needs that the first must not have lives in
``admin/static/lan.css``, and the reason that file is safe is a single property - every selector
in it is scoped ``body:not(.kiosk)``, so a rule *cannot* reach the panel however the cascade
falls out.

That is worth a test rather than a comment. The panel has no scrollbar, no zoom, no address bar
and nobody watching it start: a stray unscoped rule there is a screen that is wrong until
somebody walks over to the bench and notices. ``tests/layout_check.mjs`` would catch it, but it
needs Chromium and a running server; this needs neither and runs in a millisecond, which is what
makes it the one that actually runs.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src/cyclops/admin/static"
LAN = STATIC / "lan.css"
SCOPE = "body:not(.kiosk)"


def selectors(css: str) -> list[str]:
    """Every selector in the sheet, comments and at-rule preludes removed.

    Blocks nested in an @media are found by the same pass: the prelude is skipped by the caller
    and the rules inside it are ordinary `selector { ... }` runs like any other.

    @keyframes are cut out first, because the blocks inside one are not selectors at all - `0%`
    and `from` name a point in an animation and match no element, so scoping them would be
    meaningless and reading them as leaks is a false alarm. What actually decides whether an
    animation can reach the panel is the rule that names it, and that is an ordinary selector
    caught by the pass below.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css, flags=re.S)
    found = []
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", css):
        prelude = block.group(1).strip()
        if prelude.startswith("@"):  # an @media prelude, not a selector
            continue
        found += [one.strip() for one in prelude.split(",") if one.strip()]
    return found


def test_every_rule_written_for_the_lan_is_fenced_off_from_the_panel() -> None:
    leaked = [one for one in selectors(LAN.read_text()) if SCOPE not in one and one != ":root"]
    assert not leaked, (
        f"lan.css rules that can reach the panel: {leaked}. Scope them with {SCOPE} - the panel "
        "is the one client that cannot be looked at from here."
    )


def test_the_only_thing_lan_css_says_to_everybody_is_which_query_the_script_reads() -> None:
    """The one permitted exception, pinned so it stays the only one.

    ``--wide-q`` has to be on :root because app.js reads it off documentElement - it is the one
    definition of "is this something you sit down and read on", and having it in the stylesheet
    is what stops the query being written out twice and drifting. A custom property renders
    nothing, so it is not an exception to the rule above so much as outside it.
    """
    body = re.sub(r"/\*.*?\*/", "", LAN.read_text(), flags=re.S)
    declared = [d.strip() for d in re.findall(r":root\s*\{([^}]*)\}", body)]
    assert len(declared) == 1, f"more than one :root block in lan.css: {declared}"


def test_the_panel_keeps_its_own_unit() -> None:
    """--u means one thing on the panel and another everywhere else; both have to exist.

    Everything in these stylesheets is `calc(N * var(--u))`, so if the body.kiosk declaration
    ever goes missing the panel silently inherits the LAN unit and the whole 800x480 layout
    quietly stops being one screenful.
    """
    base = (STATIC / "base.css").read_text()
    assert re.search(r"body\.kiosk\s*\{[^}]*--u:", base), "base.css has no --u for the panel"
    assert re.search(r":root\s*\{[^}]*--u:", base, flags=re.S), "base.css has no --u for the LAN"
