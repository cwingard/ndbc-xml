#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Persist the last-processed bin timestamp across pipeline runs.

State file schema::

    {
        "last_bin_end": "2026-03-30T12:00:00+00:00",  // ISO-8601 UTC
        "station": "CE02"
    }

A missing or unreadable state file means no prior runs exist for this
deployment directory, so processing starts from the beginning of the
available data. This is the natural behavior when a new deployment
directory (e.g. ``D00021``) is passed for the first time — no
additional indicator files are needed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_state(state_file: Path) -> pd.Timestamp | None:
    """Read the last-processed bin-end timestamp from *state_file*.

    Parameters
    ----------
    state_file : Path
        Path to the JSON state file.

    Returns
    -------
    pd.Timestamp or None
        UTC timestamp of the last completed bin end, or ``None`` if
        the file does not exist or cannot be parsed (triggers a fresh
        start from the beginning of available data).
    """
    if not state_file.exists():
        log.info("No state file at %s — starting fresh.", state_file)
        return None
    try:
        data = json.loads(state_file.read_text())
        ts = pd.Timestamp(data["last_bin_end"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "Could not read state from %s: %s — starting fresh.",
            state_file, exc,
        )
        return None


def save_state(
    state_file: Path,
    last_bin_end: pd.Timestamp,
    station: str = "",
) -> None:
    """Write *last_bin_end* to *state_file*.

    Parameters
    ----------
    state_file : Path
        Destination path. Parent directory is created if absent.
    last_bin_end : pd.Timestamp
        Timestamp of the last completed bin's right edge (UTC).
    station : str
        Optional station label written for human readability.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_bin_end": last_bin_end.isoformat(),
        "station": station,
    }
    state_file.write_text(json.dumps(payload, indent=2))
    log.debug("State saved: %s → %s", station, last_bin_end.isoformat())


def clear_state(state_file: Path) -> None:
    """Delete *state_file* to force reprocessing from scratch.

    Parameters
    ----------
    state_file : Path
        Path to the JSON state file. A missing file is silently
        ignored — the effect (fresh start) is the same.
    """
    if state_file.exists():
        state_file.unlink()
        log.info("State file deleted: %s — will reprocess from scratch.",
                 state_file)
    else:
        log.info("No state file at %s — already starting fresh.", state_file)


def bin_start_from_state(
    state_file: Path,
    earliest_data_time: pd.Timestamp,
) -> pd.Timestamp:
    """Determine the first bin-edge to process.

    - If *state_file* exists and is readable, resume from the saved
      ``last_bin_end``.
    - Otherwise (new deployment directory, first ever run, or after
      ``--reprocess``) start ~1 hour after the earliest available
      data point, floored to the nearest 10-minute boundary. The
      1-hour offset guards against the edge case where the first few
      samples in a new deployment may be incomplete.

    Parameters
    ----------
    state_file : Path
        JSON state file path.
    earliest_data_time : pd.Timestamp
        Timestamp of the first available raw observation (UTC).

    Returns
    -------
    pd.Timestamp
        UTC bin-start timestamp aligned to a 10-minute boundary.
    """
    saved = load_state(state_file)
    if saved is not None:
        return saved
    return (earliest_data_time + pd.Timedelta(hours=1)).floor("10min")
