"""The standing facts about whoever Cyclops works with, and the one way this can go badly.

Everything here guards a silent failure rather than a crash, and they are not the same shape:

* **A bad round trip empties the file.** The list is rewritten rather than appended to, so the
  write path is one bad reply away from replacing four true sentences about somebody with
  nothing. ``[]`` therefore has to mean *change nothing*, everywhere, and the two tests about
  that are the reason this file exists.
* **The header comes back as a fact.** The prose at the top of ``about-you.md`` is written by
  the code and read by a person. If the reader ever picked a line of it up as a bullet, it would
  be handed to a model as something true about the user, and then rewritten as though it were.
* **The caps stop holding.** This block goes into every session's instructions and competes for
  attention with the rules that make Cyclops usable, so twenty facts must come back as twelve.

Imports only :mod:`cyclops.about`, :mod:`cyclops.card` and :mod:`cyclops.config` - no key, no
network, no camera, per the rule in pyproject.toml.
"""

from __future__ import annotations

from cyclops import about
from cyclops.config import Settings

FACTS = [
    "Goes by Marco.",
    "Embedded work by trade; comfortable with a soldering iron and a scope.",
    "Has a Bambu A1 and a small bench lathe; designs in Fusion 360.",
]


def settings_for(tmp_path):
    return Settings(api_key="", about_file=tmp_path / "about-you.md")


def test_what_is_written_is_what_is_read_back(tmp_path):
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)
    assert about.read(settings) == FACTS


def test_the_header_is_prose_and_never_a_fact(tmp_path):
    """The file explains itself to whoever opens it; none of that is true about them."""
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)
    page = settings.about_file.read_text()
    assert page.startswith("# About you")
    assert "reads this at the start of every" in page, "the explanation has to survive a rewrite"
    assert about.read(settings) == FACTS, "and not one line of it is a fact about anybody"


def test_a_missing_file_is_an_empty_list_and_not_an_error(tmp_path):
    assert about.read(settings_for(tmp_path)) == []


def test_a_hand_edited_file_still_reads(tmp_path):
    """Nobody editing this should have to know which bullet the parser wants."""
    settings = settings_for(tmp_path)
    settings.about_file.write_text("# About you\n\n* Goes by Marco.\n1. Prints in PETG.\n")
    assert about.read(settings) == ["Goes by Marco.", "Prints in PETG."]


def test_deleting_a_line_by_hand_is_how_you_correct_it(tmp_path):
    """The file is the truth, not a cache of something the code holds elsewhere."""
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)
    about.write(settings, FACTS[:1])
    assert about.read(settings) == FACTS[:1]


def test_nothing_it_writes_is_ever_zero_bytes(tmp_path):
    """Through cyclops.card, like everything else on a card that gets unplugged."""
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)
    assert settings.about_file.stat().st_size > 0
    assert list(tmp_path.glob(".*.tmp")) == [], "no scratch file survives a good write"


# ------------------------------------------------------------------ reading a reply


def test_a_tidy_reply_parses():
    assert about.parse("- Goes by Marco.\n- Prints in PETG.") == [
        "Goes by Marco.",
        "Prints in PETG.",
    ]


def test_packaging_never_costs_us_the_answer():
    """A preamble and a heading are how the job looks done, not how it looks failed."""
    answer = "Here is the updated list:\n\n# About you\n\n- Goes by Marco.\n\n* Prints in PETG.\n"
    assert about.parse(answer) == ["Goes by Marco.", "Prints in PETG."]


def test_none_is_nothing_rather_than_a_fact_called_none():
    assert about.parse("none") == []
    assert about.parse("- none") == []
    assert about.parse("- None.\n- Goes by Marco.") == ["Goes by Marco."]


def test_prose_with_no_bullets_in_it_is_not_a_list():
    """The whole safety rule: an answer we cannot read must not be read as an empty list."""
    assert about.parse("I don't have enough to go on here.") == []


def test_the_same_thing_twice_is_one_thing():
    assert about.parse("- Goes by Marco.\n- goes by marco.") == ["Goes by Marco."]


def test_the_caps_hold():
    """This block goes into every session's instructions; it competes with the rules."""
    many = "\n".join(f"- Fact number {n}." for n in range(30))
    assert len(about.parse(many)) == about.MAX_FACTS

    long = "- " + " ".join(["word"] * 200)
    only = about.parse(long)[0]
    assert len(only) <= about.MAX_FACT_CHARS
    assert only.endswith("…"), "cut at a word boundary and said so"


# ------------------------------------------------------------------ the empty answer


def test_remembering_without_a_key_changes_nothing(tmp_path):
    settings = settings_for(tmp_path)
    assert about.remember("User: hello", FACTS, settings) == []


def test_remembering_when_it_is_turned_off_changes_nothing(tmp_path):
    settings = Settings(api_key="sk-x", about_file=tmp_path / "about-you.md", remember=False)
    assert about.remember("User: I'm Marco.", FACTS, settings) == []


def test_an_empty_transcript_never_reaches_the_model(tmp_path):
    """No key is checked second; an empty conversation is not worth a request either way."""
    settings = Settings(api_key="sk-x", about_file=tmp_path / "about-you.md")
    assert about.remember("   ", FACTS, settings) == []


# ------------------------------------------------------------------ what the model is told

# These reach into cyclops.agent, which costs an import of the openai SDK and (through
# cyclops.session) of OpenCV - a fraction of a second, and worth it: the file on the card is
# only half the feature, and the half that changes what Cyclops says is this one. Nothing here
# opens a socket.


def test_a_box_that_knows_nobody_is_told_to_ask(tmp_path):
    """The empty state is the one that has to carry an instruction rather than a blank."""
    from cyclops.agent import ABOUT_UNKNOWN, _about_block

    block = _about_block(settings_for(tmp_path))
    assert block == ABOUT_UNKNOWN


def test_what_it_knows_is_handed_over_under_a_header(tmp_path):
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)

    from cyclops.agent import ABOUT_MAX_CHARS, _about_block

    block = _about_block(settings)
    assert block.startswith("WHO YOU ARE TALKING TO")
    for fact in FACTS:
        assert f"- {fact}" in block
    assert len(block) <= ABOUT_MAX_CHARS


def test_turning_it_off_says_nothing_at_all(tmp_path):
    """CYCLOPS_REMEMBER=0 is the privacy switch: no file read, and not a word in the prompt."""
    settings = Settings(api_key="", about_file=tmp_path / "about-you.md", remember=False)
    about.write(Settings(api_key="", about_file=settings.about_file), FACTS)

    from cyclops.agent import _about_block

    assert _about_block(settings) == ""


def test_a_full_card_still_fits_under_the_cap(tmp_path):
    """The cap is a backstop, not a budget: the worst case must not be trimmed mid-sentence."""
    settings = settings_for(tmp_path)
    biggest = [f"{n} " + "x" * (about.MAX_FACT_CHARS - 3) for n in range(about.MAX_FACTS)]
    about.write(settings, biggest)
    assert about.read(settings) == biggest, "the fullest list a card can hold"

    from cyclops.agent import ABOUT_MAX_CHARS, _about_block

    block = _about_block(settings)
    assert len(block) < ABOUT_MAX_CHARS
    assert block.endswith(biggest[-1] + "\n"), "nothing was cut off the end"


def test_an_empty_list_never_reaches_the_card(tmp_path):
    """Found on the Pi: a session about nothing personal left a header with no list under it.

    The rule that nothing a model says can empty this file has to live in the function that
    writes it, not only in the one caller that currently guards the call.
    """
    settings = settings_for(tmp_path)
    about.write(settings, FACTS)
    about.write(settings, [])
    assert about.read(settings) == FACTS, "the list stands"

    fresh = settings_for(tmp_path / "elsewhere")
    about.write(fresh, [])
    assert not fresh.about_file.exists(), "and a headers-only file is never created either"
