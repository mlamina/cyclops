"""Naming a finished session, so the folder on the card says what it was about.

A session's folder is created the moment it starts, when nobody yet knows what the
conversation will be - so it is named for the clock: ``2026-08-26_14-32-05``. That is already
a good name, sortable and honest. This turns it into a better one at the end, by asking a small
model what the thing was actually about: ``2026-08-26_14-32-05_lego-falcon``.

Everything here is best-effort by design. No network, no key, an empty conversation, a slow
answer, a model that replies with a paragraph - all of them land on ``""``, and the session
keeps the name it already had. A folder that is merely dated is a perfectly good outcome; a
kiosk that hangs on shutdown waiting for a name is not.
"""

from __future__ import annotations

import re
import unicodedata

from openai import APIError, OpenAI

from .config import Settings

# The same pick, for the same reason, as ``search.SEARCH_MODEL``: fastest of the current
# generation, and this is a four-word answer, not an essay. Kept as its own constant because a
# naming model and a search model happening to be the same string today is a coincidence.
SLUG_MODEL = "gpt-5.4-nano"
# Short on purpose. This runs while the kiosk is shutting down, alongside the mux; a name is
# never worth making someone wait for.
SLUG_TIMEOUT_S = 6.0
MAX_WORDS = 4
MAX_CHARS = 40
MAX_TRANSCRIPT_CHARS = 3000  # conversations name themselves in their opening minute

SLUG_PROMPT = """\
Name this conversation for a folder on a memory card. Reply with two to four lowercase words
joined by hyphens, and nothing else - no quotes, no punctuation, no date, no file extension.
Name what it was actually about, the way a person labelling the folder later would:
lego-falcon, bike-brake-torque, kitchen-tap-leak, homework-fractions.
If it was too short or too vague to be about anything, reply with: chat

Conversation:
{transcript}"""


def name_session(transcript: str, settings: Settings) -> str:
    """Two to four hyphenated words for the folder name, or ``""`` if none can be had.

    Returning ``""`` is a first-class outcome, not a failure, and this never raises: the caller
    is a teardown path that must finish either way.
    """
    transcript = transcript.strip()[:MAX_TRANSCRIPT_CHARS]
    if not transcript or not settings.slug or not settings.api_key:
        return ""  # nothing to name it from, or naming is switched off - don't call at all

    # max_retries=0 is load-bearing: the SDK retries twice by default, which would silently turn
    # a 6 s budget into 18 s of a kiosk that looks hung after you tapped stop.
    client = OpenAI(api_key=settings.api_key, timeout=SLUG_TIMEOUT_S, max_retries=0)
    try:
        response = client.responses.create(
            model=SLUG_MODEL,
            reasoning={"effort": "none"},  # a folder name does not need deliberation
            input=SLUG_PROMPT.format(transcript=transcript),
        )
        answer = (getattr(response, "output_text", "") or "").strip()
    except (APIError, OSError, ValueError):
        return ""  # offline, refused, timed out - the date-stamped name stands
    except Exception:  # noqa: BLE001 - a name is never worth failing a teardown over
        return ""
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return slugify(answer)


def slugify(text: str) -> str:
    """A model's answer, reduced to something safe to put in a filename.

    Applied unconditionally and never skipped: this is the one place a filename is built out of
    text a model chose, and a model that decides to answer with a path, a quote or an emoji must
    not be able to reach the filesystem with it.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not text:
        return ""
    text = "-".join(text.split("-")[:MAX_WORDS])[:MAX_CHARS].strip("-")
    return "" if text in {"", "chat"} else text
