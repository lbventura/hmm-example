"""Minimal FASTA parser.

Just enough to read the HMMER tutorial test files. Each record's sequence is
returned uppercase with all whitespace stripped. The first whitespace-delimited
token of the header is used as the record's name.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


def parse_fasta(stream: TextIO) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) tuples from a FASTA stream."""
    name: str | None = None
    chunks: list[str] = []

    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks).upper()
            header = line[1:].strip()
            name = header.split()[0] if header else ""
            chunks = []
        else:
            chunks.append("".join(line.split()))
    if name is not None:
        yield name, "".join(chunks).upper()


def read_first(path: str | Path) -> tuple[str, str]:
    """Read ``path`` as FASTA and return the first ``(name, sequence)`` record."""
    with open(path) as f:
        return next(parse_fasta(f))
