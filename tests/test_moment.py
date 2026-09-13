"""What a session is told about when it is starting - see :func:`cyclops.session.now_context`.

The greeting is the only thing that reads it, and the only way to check a greeting knows which
morning it is being said into is to hand it a made-up morning. So every test here fabricates a
card and a clock; nothing reads the real one, and nothing asks what the prose says.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from cyclops import card, session
from cyclops.config import Settings

NOW = datetime(2026, 9, 14, 8, 14)  # a Monday morning


@pytest.fixture
def card_dir(tmp_path):
    return tmp_path / "sessions"


def wake(card_dir, started: datetime, *, minutes: float = 12, finished: bool = True) -> None:
    """A session folder on the card, as one that ran for ``minutes`` and ended would look."""
    folder = card_dir / started.strftime(session.STAMP)
    folder.mkdir(parents=True)
    log = [
        {"type": "session", "uuid": "x"},
        {"type": "you", "text": "how deep should this go?"},
        {"type": "end", "seconds": minutes * 60},
    ]
    card.write_text(folder / session.LOG_NAME, "".join(json.dumps(r) + "\n" for r in log))
    if finished:
        card.write_text(folder / session.PAGE_NAME, "# a session\n")


def moment(card_dir, now=NOW):
    return session.now_context(Settings(api_key="", sessions_dir=card_dir), now=now)


def test_an_empty_card_says_it_has_never_been_switched_on(card_dir):
    card_dir.mkdir(parents=True)
    got = moment(card_dir)
    assert got.since is None and got.nth_today == 1


def test_the_first_wake_of_a_day_is_the_first(card_dir):
    wake(card_dir, NOW - timedelta(days=1, hours=3))
    assert moment(card_dir).nth_today == 1


def test_every_wake_after_that_counts_up(card_dir):
    for hour in (9, 11, 14, 16):
        wake(card_dir, NOW.replace(hour=hour))
    assert moment(card_dir, now=NOW.replace(hour=17)).nth_today == 5


def test_the_gap_is_measured_from_when_the_last_one_switched_off(card_dir):
    """Not from when it started: a two-hour session that ended ten minutes ago is ten minutes."""
    wake(card_dir, NOW - timedelta(hours=2, minutes=10), minutes=120)
    assert moment(card_dir).since == timedelta(minutes=10)


def test_the_folder_being_set_up_right_now_is_not_a_session_that_happened(card_dir):
    """It exists before the model is asked anything, and it has no page until it ends."""
    wake(card_dir, NOW - timedelta(hours=3))
    wake(card_dir, NOW - timedelta(seconds=1), finished=False)
    got = moment(card_dir)
    assert got.nth_today == 2 and got.since == timedelta(hours=2, minutes=48)


def test_the_same_situation_twice_reads_the_same(card_dir, tmp_path):
    """The half of the criterion a prompt cannot fake by being random."""
    other = tmp_path / "other"
    for where in (card_dir, other):
        wake(where, NOW - timedelta(days=3))
    assert moment(card_dir).text == moment(other).text


def test_different_situations_read_differently(tmp_path):
    """Monday first thing, Friday fifth wake, back after a cup of tea, and the small hours."""
    situations = {
        "monday first thing": (NOW, [NOW - timedelta(days=3)]),
        "friday fifth": (
            NOW.replace(hour=16) + timedelta(days=4),
            [NOW.replace(hour=h) + timedelta(days=4) for h in (9, 11, 13, 15)],
        ),
        "cup of tea": (NOW.replace(hour=21), [NOW.replace(hour=20)]),
        "small hours": (NOW.replace(hour=2), [NOW - timedelta(days=9)]),
    }
    texts = set()
    for name, (when, history) in situations.items():
        here = tmp_path / name.replace(" ", "-")
        for started in history:
            wake(here, started, minutes=5)
        texts.add(session.now_context(Settings(api_key="", sessions_dir=here), now=when).text)
    assert len(texts) == len(situations)
