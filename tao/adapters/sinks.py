"""
Output sinks for emitted TAO tuples.

A sink is any callable that takes a tuple dict. The adapter ships four
ready-made sinks:

    StdoutSink     — pretty-print JSON to stdout (default; useful for dev)
    ListSink       — collect into an in-memory list (useful for tests)
    JsonlSink      — append one JSON line per tuple to a file
    CallableSink   — wrap an arbitrary function

Custom sinks are just functions: `def my_sink(tuple_dict): ...`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Protocol


class Sink(Protocol):
    """Anything callable with one tuple-dict argument is a sink."""

    def __call__(self, tao_tuple: dict[str, Any]) -> None: ...


class StdoutSink:
    """Pretty-print emitted tuples to stdout. Default sink."""

    def __init__(self, indent: int | None = 2):
        self.indent = indent

    def __call__(self, tao_tuple: dict[str, Any]) -> None:
        json.dump(tao_tuple, sys.stdout, indent=self.indent, sort_keys=False)
        sys.stdout.write("\n")
        sys.stdout.flush()


class ListSink:
    """Collect emitted tuples into a list. Most useful in tests.

    Usage:
        sink = ListSink()
        configure_emitter(sink=sink, actor=...)
        # ... run decorated functions
        for tup in sink.tuples:
            ...
    """

    def __init__(self):
        self.tuples: list[dict[str, Any]] = []

    def __call__(self, tao_tuple: dict[str, Any]) -> None:
        self.tuples.append(tao_tuple)

    def clear(self) -> None:
        self.tuples.clear()

    def __len__(self) -> int:
        return len(self.tuples)


class JsonlSink:
    """Append-only JSONL file sink for production audit logs.

    Each emitted tuple is written as one canonical JSON line. The file is
    opened lazily and closed implicitly when the process exits. Concurrent
    writes from a single process are safe; cross-process safety is the
    caller's responsibility (use a queue, or a downstream log shipper).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def _ensure_open(self):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def __call__(self, tao_tuple: dict[str, Any]) -> None:
        self._ensure_open()
        # One JSON object per line. Compact but readable; not JCS-canonical.
        # Production deployments that need RFC 8785 should wrap the sink.
        self._handle.write(json.dumps(tao_tuple, separators=(",", ":")))
        self._handle.write("\n")
        self._handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class CallableSink:
    """Adapt a plain function into a sink. Mostly for clarity; you can
    pass a function directly as a sink anywhere a Sink is accepted."""

    def __init__(self, fn: Callable[[dict[str, Any]], None]):
        self.fn = fn

    def __call__(self, tao_tuple: dict[str, Any]) -> None:
        self.fn(tao_tuple)
