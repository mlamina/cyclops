"""Reading a small model's answer about a session - what it named, and what it refused to name.

No network: every test here hands :func:`cyclops.slug.parse` a reply and checks what came out.
"""

from __future__ import annotations

import pytest

from cyclops import slug

FULL = """\
slug: bike-brake-torque
# Bled the rear brake and replaced the pads.
Bled the rear brake, replaced both pads, and torqued the caliper bolts to 25 Nm.\
"""


def test_a_named_session_keeps_all_three_parts():
    described = slug.parse(FULL)
    assert described.slug == "bike-brake-torque"
    assert described.title.startswith("Bled the rear brake")
    assert "25 Nm" in described.summary
    assert not described.nothing


@pytest.mark.parametrize("word", sorted(slug.DISCARD))
def test_the_discard_word_is_a_verdict_and_never_a_folder_name(word):
    """It is read off line one alone: the model will not stop writing the other two."""
    described = slug.parse(f"slug: {word}\n# Checked the microphone.\nNo work was done.")
    assert described.nothing
    assert described.slug == "", "a folder is never named after the answer that removes it"


def test_a_reply_that_never_arrived_is_not_a_verdict():
    """Silence and "there was no session here" used to be the same empty string. Only one acts."""
    assert not slug.parse("").nothing
    assert not slug.Description().nothing


def test_slugify_only_sanitises():
    """The semantic decision moved to parse; this is the filename guard and nothing else."""
    assert slug.slugify("chat") == "chat"
    assert slug.slugify("../../etc/passwd") == "etc-passwd"
