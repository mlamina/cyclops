"""``Project Data.xlsx``: the small hard facts about a project, and how one is found again.

A project's prose says what was decided. This says the bore is 12.7 mm. Those are different
kinds of memory and they want different storage: ``Log.md`` is appended to and summarised, and a
number that goes through a summariser comes back out as "around 13". So the numbers live in a
sheet, one row each, exactly as they were spoken or read.

**A real spreadsheet, not a serialisation format.** ``Project Data.xlsx`` sits in the project's
folder next to ``README.md`` and is meant to be opened, read and corrected in Excel by the person
whose project it is - the same promise the rest of ``projects/`` makes. Two things follow, and
both are honoured below: a workbook someone has edited by hand must survive a save with its
formatting intact, and a workbook that has drifted from what we would have written - a missing
header, a column added, rows sorted - must still read.

**This module never touches the disk.** ``store.py`` is the only writer in the package and the
rule is checkable with a grep; everything here takes ``bytes`` and returns ``bytes``, so the
grep stays empty. What is on the card is ``store``'s business, what the bytes mean is this
module's, and the split is what keeps that invariant honest rather than merely stated.

The retrieval is a word-overlap score with a spelling fallback, close kin to
:func:`store.search` and deliberately not the same: project names are typed by a model, keys
arrive through speech transcription, and an exact token intersection loses "torq" against
"torque" - which is the one query a person actually asks.
"""

from __future__ import annotations

import difflib
import io
import re
from dataclasses import dataclass

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from ..slug import fold

HEADERS = ("Key", "Value", "Note")

# Excel's own limits, not ours. A title longer than 31 characters or holding one of these is
# rejected by openpyxl outright, and a workbook cannot have two sheets with the same name.
FORBIDDEN = re.compile(r"[\\/*?:\[\]]")
MAX_TITLE_CHARS = 31
FALLBACK_TAB = "Notes"

# Below this a hit is noise. Tuned so a single shared word out of a three-word query does not
# come back as an answer, while "caliper torq" still finds "Caliper bolt torque".
MIN_SCORE = 0.34
DEFAULT_LIMIT = 5

# A query token shorter than this matches too much to be worth a prefix or fuzzy comparison:
# "m8" against "m10" scores well on characters and means something else entirely.
MIN_PARTIAL = 3
CLOSE_ENOUGH = 0.75  # SequenceMatcher ratio at which two words are the same word, misheard


@dataclass(frozen=True)
class Row:
    """One remembered thing: what it is called, what it is, and why - in one tab."""

    tab: str
    key: str
    value: str
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        """The shape handed to the voice model. ``note`` is dropped when empty, not sent blank."""
        out = {"tab": self.tab, "key": self.key, "value": self.value}
        if self.note:
            out["note"] = self.note
        return out


def sheet_title(name: str) -> str:
    """A model's tab name, reduced to something Excel will actually accept as a sheet title.

    The :func:`cyclops.slug.safe_folder_name` of this module, and for the same reason: a name a
    person reads is the point, so capitals, spaces and non-ASCII letters stay and only what the
    format forbids is removed. An empty result gets a name rather than raising, because losing a
    value over an unsayable tab name would be the worse failure.
    """
    cleaned = " ".join(FORBIDDEN.sub(" ", name).split())
    cleaned = cleaned.strip("'")  # Excel rejects a title that starts or ends with an apostrophe
    return cleaned[:MAX_TITLE_CHARS].strip() or FALLBACK_TAB


class Book:
    """One project's workbook, in memory: ordered tabs, each an ordered list of :class:`Row`.

    Ordered because the sheet is read by a person. New tabs go on the end and new keys go under
    the last one, so the file grows the way a notebook does rather than being re-sorted under
    them every time the agent writes.
    """

    def __init__(self) -> None:
        self._tabs: dict[str, list[Row]] = {}

    # -------------------------------------------------------------- reading

    def tabs(self) -> dict[str, int]:
        """``{tab: how many rows}`` - the whole summary ``open_project`` hands the model.

        This is the feature's efficiency claim in one method: a project with two hundred values
        costs the conversation two tab names and two numbers, and every value itself is a tool
        call away.
        """
        return {name: len(rows) for name, rows in self._tabs.items()}

    def rows(self) -> list[Row]:
        return [row for rows in self._tabs.values() for row in rows]

    def tab_named(self, tab: str) -> str | None:
        """The tab someone means by ``tab``, folded - so "Torque specs" cannot become two."""
        wanted = fold(tab)
        for name in self._tabs:
            if fold(name) == wanted:
                return name
        return None

    def find_key(self, key: str, tab: str | None = None) -> list[Row]:
        """Every row that could be the one called ``key``, best first.

        Exact folded matches are returned alone when there are any: a key spelled exactly right
        is never ambiguous with something that merely resembles it. Only when nothing matches
        exactly does this fall back to the fuzzy score, and then the caller has to decide - which
        for a delete means asking rather than guessing.
        """
        candidates = self._tabs.get(tab, []) if tab else self.rows()
        if tab and tab not in self._tabs:
            resolved = self.tab_named(tab)
            candidates = self._tabs.get(resolved, []) if resolved else []
        wanted = fold(key)
        if exact := [row for row in candidates if fold(row.key) == wanted]:
            return exact
        scored = [(score(key, row), row) for row in candidates]
        hits = [(value, row) for value, row in scored if value >= MIN_SCORE]
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in hits]

    # -------------------------------------------------------------- writing

    def put(self, tab: str, key: str, value: str, note: str = "") -> bool:
        """Write one pair down. Returns True if it replaced a value that was already there.

        An upsert rather than an append, because the second time you measure something is the
        time you were right. Replacing keeps the row where it was: a corrected value that jumped
        to the bottom of the sheet would read as a new fact rather than a fixed one.
        """
        name = self.tab_named(tab) or sheet_title(tab)
        rows = self._tabs.setdefault(name, [])
        wanted = fold(key)
        for i, row in enumerate(rows):
            if fold(row.key) == wanted:
                rows[i] = Row(name, row.key, value, note or row.note)
                return True
        rows.append(Row(name, key, value, note))
        return False

    def drop(self, row: Row) -> bool:
        """Remove exactly the row given. An emptied tab goes with it - a blank sheet is litter."""
        rows = self._tabs.get(row.tab)
        if rows is None or row not in rows:
            return False
        rows.remove(row)
        if not rows:
            del self._tabs[row.tab]
        return True


# ------------------------------------------------------------------ the file


def _cell(value: object) -> str:
    """One cell as text. Excel hands back numbers, dates and None; the sheet is read, not summed."""
    if value is None:
        return ""
    return str(value).strip()


def load(blob: bytes | None) -> Book:
    """Parse a workbook. A missing, empty or unreadable file is an empty book, never an error.

    Unreadable rather than raising because this is a file a person is invited to edit: the answer
    to "they saved it as something else" is that the agent starts a fresh sheet and says nothing
    was written down, not that a voice conversation ends in a traceback.
    """
    book = Book()
    if not blob:
        return book
    try:
        workbook = load_workbook(io.BytesIO(blob), data_only=True)
    except Exception:  # noqa: BLE001 - any malformed zip, and there are many kinds
        return book
    for sheet in workbook.worksheets:
        rows: list[Row] = []
        for index, cells in enumerate(sheet.iter_rows(values_only=True)):
            padded = list(cells) + [None, None, None]
            key, value, note = (_cell(padded[0]), _cell(padded[1]), _cell(padded[2]))
            # The header, when there is one. Skipped by what it says rather than by position, so
            # a sheet someone stripped the header off still reads every row it has.
            if index == 0 and fold(key) == "key":
                continue
            if key:
                rows.append(Row(sheet.title, key, value, note))
        # Registered even with nothing in it. An empty sheet here is one a person made and has
        # not filled in yet, and :func:`dump` deletes any tab this book has forgotten - so
        # dropping it on read would quietly delete their sheet on the agent's next save.
        book._tabs[sheet.title] = rows
    return book


def dump(book: Book, blob: bytes | None = None) -> bytes:
    """Serialise ``book``, editing the workbook in ``blob`` in place when there is one.

    In place, not rebuilt from scratch, because this file belongs to the person whose project it
    is: a bold header, a widened column, a fourth column of their own must still be there after
    the agent writes a value. openpyxl round-trips styles and column widths; it does not
    round-trip charts or images, and a sheet holding one is not what this file is for.
    """
    workbook = None
    if blob:
        try:
            workbook = load_workbook(io.BytesIO(blob))
        except Exception:  # noqa: BLE001 - a file we cannot read is one we replace
            workbook = None
    if workbook is None:
        workbook = Workbook()
        workbook.remove(workbook.active)

    wanted = book.tabs()
    for stale in [sheet for sheet in workbook.sheetnames if sheet not in wanted]:
        workbook.remove(workbook[stale])

    for tab, rows in book._tabs.items():
        sheet = workbook[tab] if tab in workbook.sheetnames else workbook.create_sheet(tab)
        for column, header in enumerate(HEADERS, start=1):
            cell = sheet.cell(row=1, column=column, value=header)
            if not cell.font.bold:
                cell.font = Font(bold=True)
            if sheet.column_dimensions[get_column_letter(column)].width in (None, 0):
                sheet.column_dimensions[get_column_letter(column)].width = 34 if column < 3 else 48
        # Assigned through the cell rather than passed as ``value=``: openpyxl's ``cell()``
        # ignores a None it is handed, so the second form silently cannot blank anything - which
        # is exactly what the loop below has to do.
        for index, row in enumerate(rows, start=2):
            sheet.cell(row=index, column=1).value = row.key
            sheet.cell(row=index, column=2).value = row.value
            sheet.cell(row=index, column=3).value = row.note or None
        # Rows a delete left behind. Only our three columns are cleared: a fourth column someone
        # added is theirs, and blanking it would be this file editing their work.
        for index in range(len(rows) + 2, sheet.max_row + 1):
            for column in range(1, len(HEADERS) + 1):
                sheet.cell(row=index, column=column).value = None

    if not workbook.sheetnames:  # openpyxl will not save a workbook with no sheets at all
        workbook.create_sheet(FALLBACK_TAB).append(list(HEADERS))

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# ------------------------------------------------------------------ finding one again


def _best(token: str, words: list[str], strong: bool) -> float:
    """How well one query word matches one row's words, on the scale the table in the plan sets."""
    if not words:
        return 0.0
    if token in words:
        return 1.0 if strong else 0.5
    if len(token) >= MIN_PARTIAL:
        for word in words:
            if word.startswith(token) or (len(word) >= MIN_PARTIAL and token.startswith(word)):
                return 0.8 if strong else 0.3
    if strong:
        ratio = max(difflib.SequenceMatcher(None, token, word).ratio() for word in words)
        if ratio >= CLOSE_ENOUGH:
            return 0.6
    return 0.0


def score(query: str, row: Row) -> float:
    """How well ``row`` answers ``query``. Deterministic - no model, no index, no library.

    The key carries the weight and the tab, value and note are a weaker haystack behind it, so
    "torque" finds the row called *Caliper bolt torque* rather than every row filed under
    *Torque specs*. The spelling fallback is what makes this usable through a microphone.
    """
    wanted = fold(query).split()
    if not wanted:
        return 0.0
    key = fold(row.key)
    key_words = key.split()
    rest = fold(f"{row.tab} {row.value} {row.note}").split()
    if key == " ".join(wanted):
        return 2.0  # an exact key, and nothing that merely resembles one outranks it
    total = sum(max(_best(t, key_words, True), _best(t, rest, False)) for t in wanted)
    hit = total / len(wanted)
    if " ".join(wanted) in key:
        hit += 0.25
    return hit


def search(book: Book, query: str, limit: int = DEFAULT_LIMIT) -> list[Row]:
    """The rows that answer ``query``, best first. Empty when nothing does, which is an answer."""
    scored = [(score(query, row), row) for row in book.rows()]
    hits = [(value, row) for value, row in scored if value >= MIN_SCORE]
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in hits[:limit]]
