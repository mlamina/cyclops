"""Reading a tool call's arguments while they are still being typed.

A model writes its arguments one token at a time, and the whole point of a live surface is to
act on them before the closing brace arrives. A JSON parser cannot help with that - a string
that stops mid-word is not JSON and never will be - so this reads one key's value out of the
prefix by hand, and says whether the value has finished.

There is exactly one thing here that has to be right: JSON escapes. The wire carries ``\\n``
as two characters, and a model writing Python or a list of marks writes a great many newlines.
Reading them literally gives you a single line containing backslash-n, which compiles to
nothing and parses to nothing, and the surface stays empty for the whole of the call.

Both callers want the same read and do different things with ``closed``. :mod:`cyclops.point`
drops the last line when it is False, because its line language was chosen so a half-written
line is simply the last one. :mod:`cyclops.sketch` keeps everything and lets the compiler
decide, because half-written Python does not compile and is dropped a step later anyway.
"""

from __future__ import annotations

# What a JSON string escape means. Anything else after a backslash is passed through as itself,
# which is wrong for \\uXXXX and right for everything a model actually writes.
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def value(prefix: str, key: str) -> tuple[str, bool]:
    """The value of ``key`` in a half-written JSON object, and whether it has ended.

    Returns ``("", False)`` when the key has not been reached yet, which a caller can treat the
    same as an empty value: there is nothing to show either way.
    """
    at = prefix.find(f'"{key}"')
    if at < 0:
        return "", False
    start = prefix.find('"', at + len(key) + 2)
    if start < 0:
        return "", False
    body: list[str] = []
    i = start + 1
    while i < len(prefix):
        char = prefix[i]
        if char == "\\" and i + 1 < len(prefix):
            body.append(ESCAPES.get(prefix[i + 1], prefix[i + 1]))
            i += 2
            continue
        if char == '"':
            return "".join(body), True
        body.append(char)
        i += 1
    return "".join(body), False
