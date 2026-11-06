#!/usr/bin/env python3
"""hq#599: how long can a hostile PDF hold the ledger's lock? jobs._hand_in parses under that lock.

Builds PDFs a job could leave as its result, all within the 128 MB limit, and times the exact call
_hand_in makes (reading.page_count) in a child process with a time limit and an address-space limit
where the OS enforces one (Linux). Writes only to a temp folder, makes no network or model call.

Usage: hq_599_slow_pdf.py [--max-mb 128] [--limit-s 120] [--as-gb 4] [shape ...]
"""

from __future__ import annotations

import argparse
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

HEAD = (
    b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
)
TAIL = b"trailer<</Root 1 0 R>>\n"  # no xref at all, so opening it means repairing it


def fill(path: Path, unit, size: int) -> None:
    with path.open("wb") as out:
        out.write(HEAD)
        n = 10
        while out.tell() < size - 64:
            out.write(unit(n))
            n += 1
        out.write(TAIL)


def many_objects(path: Path, size: int) -> None:
    fill(path, lambda n: b"%d 0 obj<</A 1>>endobj\n" % n, size)


def many_small_objstm(path: Path, size: int) -> None:
    body = zlib.compress(b"0 0 <<>> ")

    def unit(n: int) -> bytes:
        return (
            b"%d 0 obj<</Type/ObjStm/N 1/First 4/Filter/FlateDecode/Length %d>>stream\n%s\nendstream endobj\n"
            % (
                n,
                len(body),
                body,
            )
        )

    fill(path, unit, size)


def objstm_bomb(path: Path, size: int) -> None:
    """One object stream that inflates about 1000 times over."""
    packer = zlib.compressobj(9)
    chunk = b"0" * (1024 * 1024)
    blob = bytearray()
    while len(blob) < size - 2048:
        blob += packer.compress(chunk)
    blob += packer.flush()
    with path.open("wb") as out:
        out.write(HEAD)
        out.write(
            b"10 0 obj<</Type/ObjStm/N 1/First 4/Filter/FlateDecode/Length %d>>stream\n" % len(blob)
        )
        out.write(blob)
        out.write(b"\nendstream endobj\n")
        out.write(TAIL)


def deep_nesting(path: Path, size: int) -> None:
    with path.open("wb") as out:
        out.write(HEAD)
        out.write(b"10 0 obj\n")
        for _ in range(size // (1024 * 1024)):
            out.write(b"[" * (1024 * 1024))
        out.write(b"\nendobj\n")
        out.write(TAIL)


def endless_tree(path: Path, size: int) -> None:
    """A /Pages chain as long as the size allows, with a /Count that lies."""

    def unit(n: int) -> bytes:
        return b"%d 0 obj<</Type/Pages/Parent %d 0 R/Kids[%d 0 R]/Count 1>>endobj\n" % (
            n,
            n - 1,
            n + 1,
        )

    fill(path, unit, size)


SHAPES = {
    f.__name__: f
    for f in (many_objects, many_small_objstm, objstm_bomb, deep_nesting, endless_tree)
}
CHILD = (
    "import resource,sys,time\n"
    "gb=int(sys.argv[2])\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS,(gb*1024**3,gb*1024**3))\n"
    "except (ValueError,OSError):\n"
    "    pass\n"
    "from pathlib import Path\n"
    "from cyclops import reading\n"
    "t=time.monotonic()\n"
    "pages=reading.page_count(Path(sys.argv[1]))\n"
    "print(f'pages={pages} parse_s={time.monotonic()-t:.1f} peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=int, default=128)
    parser.add_argument("--limit-s", type=int, default=120)
    parser.add_argument("--as-gb", type=int, default=4)
    parser.add_argument("shape", nargs="*", default=list(SHAPES))
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix="qa-599-pdf-"))
    print(f"size {args.max_mb} MB, limit {args.limit_s} s, address space {args.as_gb} GB")
    try:
        for name in args.shape:
            path = folder / f"{name}.pdf"
            SHAPES[name](path, args.max_mb * 1024 * 1024)
            started = time.monotonic()
            try:
                out = subprocess.run(
                    [sys.executable, "-c", CHILD, str(path), str(args.as_gb)],
                    capture_output=True,
                    text=True,
                    timeout=args.limit_s,
                    check=False,
                )
                tail = (
                    out.stdout.strip()
                    or f"no output, exit {out.returncode}: {out.stderr.strip()[-160:]}"
                )
                print(f"{name}: {tail} wall_s={time.monotonic() - started:.1f}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"{name}: STILL PARSING at the {args.limit_s} s limit, killed", flush=True)
            path.unlink()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(f"children peak_rss={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
