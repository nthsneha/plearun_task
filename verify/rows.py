"""The one shape every record line is turned into, whatever file it came from."""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

from verify.timeparse import Stamp


@dataclass
class Row:
    file: str          # record file name, e.g. "camera_log.csv" (what we cite)
    row: int           # number we cite: CSV data row, JSON event or VTT cue, counting from 1
    raw: str           # the row's text exactly as it appears in the file (quotes are checked against this)
    fmt: str           # "csv", "json" or "vtt"
    device: str        # whose clock stamped this row: a "device" column if present, else the file name
    fields: Dict[str, str] = field(default_factory=dict)  # column/key -> value, all as text
    text: str = ""     # the free-text content (note text, cue words), for keyword matching
    time_text: Optional[str] = None  # the timestamp exactly as written, before parsing
    time: Optional[Stamp] = None     # filled in by loaders.assign_times
    events: Optional[FrozenSet[str]] = None  # cache for clocks.row_events (worked out once per row)
