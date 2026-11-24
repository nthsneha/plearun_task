"""Work out how far each device's clock is off, using events seen by more than one device.

Words used here:
  device     whose clock stamped a row: CAM-07, dispatch.json, notes..., a transcript
  anchor     an event a device recorded: a stop, a 911 call, a unit on scene
  offset     what to ADD to a device's time to get reference time
  reference  the one clock every other clock is measured against (offset exactly 0)

Every offset is a RANGE (lo, hi), never a single guess. If the notes say
"21:11" (any second in that minute) and dispatch says 21:10:19 for the same
call, the dispatch offset is somewhere from +41s to +101s.

Rules that keep this careful:
  1. Only record rows are used. The report's claims are never anchors.
  2. A device that recorded the same event at clearly different times is
     ambiguous for that event, so the event is not used for that device.
     Exact duplicates (same time, repeated row) count once.
  3. All anchors linking a device to aligned devices must agree. If they
     contradict each other, the device is left unaligned.
  4. An offset that is very wide, or implausibly large for clock drift
     (e.g. a wrong timezone guess), leaves the device unaligned.
Rows from an unaligned device are never used to judge a time.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from verify.events import events_in
from verify.rows import Row

Range = Tuple[float, float]

ANCHOR_SLACK = 15.0          # seconds: two sources describing one event may differ a little
DUPLICATE_WINDOW = 5.0       # seconds: same event this close on one device is a repeat, not a new event
MAX_OFFSET_WIDTH = 300.0     # seconds: a vaguer offset than this is useless, so do not align
MAX_ABS_OFFSET = 15 * 60.0   # seconds: real clock drift is minutes, not hours (hours = wrong timezone)

_EVENT_FIELD_RE = re.compile(r"^(?:type|event|event_type|kind|status|code)$", re.I)
_SPEED_FIELD_RE = re.compile(r"speed", re.I)


@dataclass
class Anchor:
    device: str
    event: str
    lo: float
    hi: float
    row: Row


@dataclass
class Alignment:
    reference: Optional[str]
    offsets: Dict[str, Range]          # aligned devices only
    reasons: Dict[str, str]            # device -> why it is or is not aligned (for the README/debugging)

    def is_aligned(self, device: str) -> bool:
        return device in self.offsets

    def to_reference(self, row: Row) -> Optional[Range]:
        """The row's time on the reference clock, as a range. None if it cannot be placed."""
        if row.time is None or row.device not in self.offsets:
            return None
        off_lo, off_hi = self.offsets[row.device]
        return (row.time.lo + off_lo, row.time.hi + off_hi)


@lru_cache(maxsize=None)
def is_event_key(key: str) -> bool:
    return bool(_EVENT_FIELD_RE.match(key))


@lru_cache(maxsize=None)
def is_speed_key(key: str) -> bool:
    return bool(_SPEED_FIELD_RE.search(key))


def row_events(row: Row) -> frozenset:
    """Which event this row records at its own timestamp. Worked out once per row, then reused.

    Structured event fields (dispatch "type") are used first. A camera
    reading of 0 mph counts as a stop. Otherwise the row's free text is read.
    """
    if row.events is None:
        row.events = frozenset(_compute_row_events(row))
    return row.events


def _compute_row_events(row: Row) -> set:
    for key, value in row.fields.items():
        if is_event_key(key):
            return events_in(value)
    for key, value in row.fields.items():
        if is_speed_key(key):
            return {"stop"} if value.strip() in ("0", "0.0") else set()
    return events_in(row.text)


def find_anchors(rows: List[Row]) -> List[Anchor]:
    """One anchor per (device, event), only where that is unambiguous."""
    grouped: Dict[Tuple[str, str], List[Anchor]] = {}
    for row in rows:
        if row.time is None:
            continue
        events = row_events(row)
        if len(events) != 1:
            continue  # no event, or a row that mentions several: not a clean anchor
        event = next(iter(events))
        grouped.setdefault((row.device, event), []).append(
            Anchor(row.device, event, row.time.lo, row.time.hi, row))

    anchors = []
    for group in grouped.values():
        group.sort(key=lambda a: a.lo)
        if group[-1].lo - group[0].lo <= DUPLICATE_WINDOW:  # all repeats of one moment
            first = group[0]
            anchors.append(Anchor(first.device, first.event, first.lo,
                                  max(a.hi for a in group), first.row))
        # else: the device saw this "event" at different times, so it is ambiguous; skip it
    return anchors


def _pair_offset(known: Anchor, other: Anchor) -> Range:
    """Range of (other's offset - known's offset) implied by one shared event."""
    return (known.lo - other.hi - ANCHOR_SLACK, known.hi - other.lo + ANCHOR_SLACK)


def _choose_reference(anchors: List[Anchor]) -> Optional[str]:
    """The absolute-time device linked by shared events to the most other anchors."""
    links: Dict[str, int] = {}
    relative = {a.device for a in anchors if a.row.time.relative}
    for a in anchors:
        links.setdefault(a.device, 0)
        for b in anchors:
            if a.event == b.event and a.device != b.device:
                links[a.device] += 1
    candidates = [d for d in links if d not in relative]
    if not candidates:
        return None
    return max(sorted(candidates), key=lambda d: links[d])


def align(rows: List[Row]) -> Alignment:
    anchors = find_anchors(rows)
    reference = _choose_reference(anchors)
    devices = sorted({r.device for r in rows})
    reasons = {d: "no event shared with an aligned device" for d in devices}
    if reference is None:
        return Alignment(None, {}, reasons)

    offsets: Dict[str, Range] = {reference: (0.0, 0.0)}
    reasons[reference] = "reference clock"
    rejected = set()
    by_device: Dict[str, List[Anchor]] = {}
    for a in anchors:
        by_device.setdefault(a.device, []).append(a)

    changed = True
    while changed:  # keep going while each pass aligns at least one more device
        changed = False
        for device in devices:
            if device in offsets or device in rejected:
                continue
            implied = []
            for mine in by_device.get(device, []):
                for known in by_device_aligned(by_device, offsets, mine.event, device):
                    k_lo, k_hi = offsets[known.device]
                    p_lo, p_hi = _pair_offset(known, mine)
                    implied.append((k_lo + p_lo, k_hi + p_hi, mine.event))
            if not implied:
                continue
            lo = max(i[0] for i in implied)
            hi = min(i[1] for i in implied)
            events = sorted({i[2] for i in implied})
            relative = by_device[device][0].row.time.relative
            if lo > hi:
                rejected.add(device)
                reasons[device] = f"anchors disagree ({', '.join(events)})"
            elif hi - lo > MAX_OFFSET_WIDTH:
                rejected.add(device)
                reasons[device] = "offset too uncertain"
            elif not relative and max(abs(lo), abs(hi)) > MAX_ABS_OFFSET:
                rejected.add(device)
                reasons[device] = "offset implausibly large (timezone problem?)"
            else:
                offsets[device] = (lo, hi)
                reasons[device] = f"aligned from {', '.join(events)}"
                changed = True

    # Final check: a device aligned early was only compared with devices aligned
    # before it. Re-check every pair now, and drop any device that contradicts another.
    changed = True
    while changed:
        changed = False
        for device in sorted(offsets):
            if device == reference:
                continue
            lo, hi = offsets[device]
            for mine in by_device.get(device, []):
                for known in by_device_aligned(by_device, offsets, mine.event, device):
                    k_lo, k_hi = offsets[known.device]
                    p_lo, p_hi = _pair_offset(known, mine)
                    lo, hi = max(lo, k_lo + p_lo), min(hi, k_hi + p_hi)
            if lo > hi:
                del offsets[device]
                reasons[device] = "anchors disagree with another aligned device"
                changed = True
                break
    return Alignment(reference, offsets, reasons)


def by_device_aligned(by_device, offsets, event, exclude):
    """Anchors for this event on devices that are already aligned."""
    return [a for d, items in by_device.items() if d in offsets and d != exclude
            for a in items if a.event == event]
