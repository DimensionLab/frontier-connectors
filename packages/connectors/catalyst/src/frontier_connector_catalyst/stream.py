# SPDX-License-Identifier: Apache-2.0
"""What every in-situ step source has in common: the ``Step`` record and the follow options."""

from __future__ import annotations

import time as _time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from frontier_connectors.errors import ConnectorError

#: names of the options understood by ``read`` / ``describe`` / the importer
FOLLOW_OPTIONS = (
    "follow",
    "timeout",
    "poll",
    "until",
    "on_timeout",
    "on_restart",
    "order_grace",
    "partial_grace",
    "settle",
    "max_steps",
)


class StreamTimeoutError(ConnectorError):
    """No new step arrived within ``timeout`` seconds and ``on_timeout=error`` was asked for
    (or nothing arrived at all)."""


@dataclass
class Step:
    """One complete step of a stream."""

    #: number the producer gave the step (file index, cycle, ADIOS step)
    index: int
    tree: dict[str, Any]
    time: float | None = None
    cycle: int | None = None
    #: index of the step the producer emitted before this one (``None``: first or unknown)
    prev: int | None = None
    run_id: str | None = None
    #: False when the producer omitted the mesh (static-mesh streams)
    has_mesh: bool = True


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _seconds(options: Mapping[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise ConnectorError(
            f"bad option {key}={value!r}", hint=f"{key} is a number of seconds"
        ) from e
    if out < 0:
        raise ConnectorError(f"bad option {key}={value!r}", hint=f"{key} must be >= 0")
    return out


@dataclass
class FollowOptions:
    """How a live stream is followed.

    * ``follow``: keep waiting for new steps (default ``False``: read what is there now);
    * ``timeout``: seconds without a new complete step before the follower gives up;
    * ``poll``: seconds between looks at the source;
    * ``until``: ``"done"`` stops at the producer's end marker (or at the timeout),
      ``"timeout"`` ignores end markers and runs until nothing arrived for ``timeout`` seconds;
    * ``on_timeout``: ``"stop"`` ends the stream normally (the dataset keeps the steps that
      arrived; this is what a crashed solver looks like), ``"error"`` raises
      :class:`StreamTimeoutError`. A stream that never delivered a step always raises;
    * ``on_restart``: ``"skip"`` ignores steps the producer re-emits after a restart from a
      checkpoint (they are already in the dataset), ``"error"`` raises;
    * ``order_grace``: seconds to wait for a step that the producer's ``prev`` chain says is
      missing before going on without it;
    * ``partial_grace``: seconds a visible but unreadable step file may stay unchanged before
      it is declared corrupt (covers writers that do not rename atomically);
    * ``settle``: seconds a file written without atomic rename (Conduit relay output) must be
      unmodified before it is read;
    * ``max_steps``: stop after this many steps.
    """

    follow: bool = False
    timeout: float = 60.0
    poll: float = 0.2
    until: str = "done"
    on_timeout: str = "stop"
    on_restart: str = "skip"
    order_grace: float = 2.0
    partial_grace: float = 5.0
    settle: float = 0.5
    max_steps: int | None = None
    #: injectable for tests
    clock: Callable[[], float] = field(default=_time.monotonic, repr=False)
    sleep: Callable[[float], None] = field(default=_time.sleep, repr=False)

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> FollowOptions:
        o = dict(options or {})
        out = cls(
            follow=_truthy(o.get("follow", False)),
            timeout=_seconds(o, "timeout", 60.0),
            poll=max(_seconds(o, "poll", 0.2), 0.001),
            until=str(o.get("until", "done")).replace("-marker", ""),
            on_timeout=str(o.get("on_timeout", "stop")),
            on_restart=str(o.get("on_restart", "skip")),
            order_grace=_seconds(o, "order_grace", 2.0),
            partial_grace=_seconds(o, "partial_grace", 5.0),
            settle=_seconds(o, "settle", 0.5),
            max_steps=int(o["max_steps"]) if o.get("max_steps") is not None else None,
        )
        if out.until not in ("done", "timeout"):
            raise ConnectorError(
                f"bad option until={o.get('until')!r}", hint="use until=done-marker | timeout"
            )
        if out.on_timeout not in ("stop", "error"):
            raise ConnectorError(
                f"bad option on_timeout={out.on_timeout!r}", hint="use on_timeout=stop | error"
            )
        if out.on_restart not in ("skip", "error"):
            raise ConnectorError(
                f"bad option on_restart={out.on_restart!r}", hint="use on_restart=skip | error"
            )
        return out
