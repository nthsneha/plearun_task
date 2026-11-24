"""Decide each claim: match, conflict, not_settled or not_checkable.

This is plain rule-based code, with no model and no randomness: the same
inputs always give the same output. The rules lean towards "not_settled",
because a wrong conflict costs far more than a cautious answer.

Guardrails:
  G1  Careful ranges. A row's time on the reference clock is a range (its own
      precision plus its clock-offset range). A verdict is given only if it
      holds for every point in that range.
  G2  Two margins. "match" needs the evidence inside a small margin around the
      claim. "conflict" needs it outside a much larger one. In between is not_settled.
  G3  Rows from unaligned devices are never used to judge a time.
  G4  A row whose timezone we had to assume, on a device whose other rows DO state
      a timezone (dispatch's "2026-03-02 21:10:19"), may support a match, never a
      conflict: that one row could be hours off. A device that never states a
      timezone (bare "21:15" notes) is consistent with itself, so G4 does not apply.
  G5  Rows that cannot identify the vehicle (camera rows have no plate/vehicle
      field) may support a match, never a conflict.
  G6  Evidence that disagrees is not_settled. A clear match is cancelled by any
      row that clearly points elsewhere (a near miss or a far row), but not by a
      row that is merely vague. A conflict needs at least one row that may conflict,
      and every other row must be far off too. Rows barred by G4/G5 can join a
      conflict, but can never cause one alone.
  G7  Every finding goes through quote.make_finding, which checks the cite
      and the exact quote.
Claim times are read on the reference clock (see clocks.py).
"""
import re
from datetime import date, timezone
from functools import lru_cache
from typing import List, Optional, Tuple

from verify.claims import Claim, Fact
from verify.clocks import Alignment, row_events
from verify.quote import make_finding, quote_for
from verify.rows import Row
from verify.timeparse import parse_timestamp

Range = Tuple[float, float]

# Seconds around the claimed minute.       (no hedge, hedged)
MATCH_SLACK = {False: 30.0, True: 90.0}
CONFLICT_SLACK = {False: 180.0, True: 300.0}
# Minutes around a claimed gap ("2 minutes after the stop").
GAP_MATCH_TOL = {False: 0.5, True: 1.0}
GAP_CONFLICT_TOL = {False: 2.0, True: 3.0}
# mph around a claimed speed. Speeds are only ever matched, never conflicted (G5).
SPEED_MATCH_TOL = {False: 1.0, True: 3.0}

_SPEED_FIELD = re.compile(r"speed", re.I)
_ZONE_FIELD = re.compile(r"zone|limit|posted", re.I)
_IDENTITY_FIELD = re.compile(r"plate|vehicle|vin|tag", re.I)
_EVENT_FIELD = re.compile(r"^(?:type|event|event_type|kind|status|code)$", re.I)

MATCH, CONFLICT, UNSURE, NOT_CHECKABLE = "match", "conflict", "not_settled", "not_checkable"
# Internal only, never written to output:
MISS = "near_miss"  # wholly outside the match window, but inside the conflict window
FAR = "far_barred"  # wholly outside the conflict window, but not allowed to conflict (G4/G5)


class Context:
    """Everything the judge needs, passed around in one place."""

    def __init__(self, rows: List[Row], alignment: Alignment, tz: timezone, base_date: date):
        self.rows = rows
        self.alignment = alignment
        self.tz = tz
        self.base_date = base_date
        self.known = {(r.file, r.row) for r in rows}
        # Devices with at least one row that states its timezone (G4 applies to these).
        self.states_tz = {r.device for r in rows if r.time and not r.time.relative and not r.time.assumed_tz}

    def claim_range(self, clock: str) -> Optional[Range]:
        st = parse_timestamp(clock, self.tz, self.base_date)
        return (st.lo, st.hi) if st else None


# ------------------------------------------------------------ small helpers

@lru_cache(maxsize=None)
def _is_speed_key(key: str) -> bool:
    return bool(_SPEED_FIELD.search(key)) and not _is_zone_key(key)


@lru_cache(maxsize=None)
def _is_zone_key(key: str) -> bool:
    return bool(_ZONE_FIELD.search(key))


def _speed_field(row: Row) -> Optional[str]:
    return next((k for k in row.fields if _is_speed_key(k)), None)


def _zone_field(row: Row) -> Optional[str]:
    return next((k for k in row.fields if _is_zone_key(k)), None)


def _number(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _identifies_vehicle(row: Row) -> bool:
    """False for rows like camera readings, which could be any car (G5)."""
    if _speed_field(row) is None:
        return True  # not a sensor reading: notes, dispatch events, transcripts
    return any(_IDENTITY_FIELD.search(k) and v.strip() for k, v in row.fields.items())


def _can_conflict(row: Row, ctx: "Context") -> bool:
    guessed_tz = bool(row.time and row.time.assumed_tz and row.device in ctx.states_tz)  # G4
    return not guessed_tz and _identifies_vehicle(row)  # G5


def _evidence_rank(row: Row, rng: Range):
    """Which supporting row to cite: structured event rows first, then the most
    precise time, then file/row order (so the choice never depends on chance)."""
    structured = any(_EVENT_FIELD.match(k) for k in row.fields)
    return (not structured, not _identifies_vehicle(row), rng[1] - rng[0], row.file, row.row)


def _inside(inner: Range, outer: Range) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _apart(a: Range, b: Range) -> bool:
    return a[1] < b[0] or b[1] < a[0]


def _classify(rng: Range, target: Range, match_slack: float, conflict_slack: float, can_conflict: bool) -> str:
    """How one row's range relates to the claim (G1, G2).
      MATCH     wholly inside the small match window
      CONFLICT  wholly outside the big conflict window (and allowed to conflict, G4/G5)
      FAR       wholly outside the big conflict window, but not allowed to conflict
      MISS      wholly outside the match window, inside the conflict window
      UNSURE    partly inside the match window: too vague to say"""
    if _inside(rng, (target[0] - match_slack, target[1] + match_slack)):
        return MATCH
    if _apart(rng, (target[0] - conflict_slack, target[1] + conflict_slack)):
        return CONFLICT if can_conflict else FAR
    if _apart(rng, (target[0] - match_slack, target[1] + match_slack)):
        return MISS
    return UNSURE


def _combine(per_row: List[Tuple[str, Row, Range]]):
    """G6: put the per-row results together into one verdict."""
    kinds = {v for v, _, _ in per_row}
    if MATCH in kinds and not kinds & {CONFLICT, MISS, FAR}:
        verdict = MATCH                 # vague rows do not cancel a clear match
    elif CONFLICT in kinds and kinds <= {CONFLICT, FAR}:
        verdict = CONFLICT              # a reliable row conflicts, and every other row is far off too
    else:
        return UNSURE, None
    best = min((t for t in per_row if t[0] == verdict), key=lambda t: _evidence_rank(t[1], t[2]))
    return verdict, best[1]


def _time_quote(row: Row) -> str:
    return quote_for(row, [row.time_text] if row.time_text else [])


# ------------------------------------------------------------ one fact each

def judge_time(fact: Fact, ctx: Context):
    """'<event> at <clock>': compare every aligned row that records that event."""
    rows = [r for r in ctx.rows if fact.event and row_events(r) == {fact.event}]
    if not rows:
        return NOT_CHECKABLE, None, None
    return time_verdict(fact, rows, ctx)


def time_verdict(fact: Fact, rows: List[Row], ctx: Context, allow_conflict: bool = True):
    """Judge a time fact against the given rows, with guardrails G1-G6.
    allow_conflict=False lets a caller use rows as possible support only."""
    claim = ctx.claim_range(fact.clock)
    if claim is None:
        return UNSURE, None, None
    m, c = MATCH_SLACK[fact.hedged], CONFLICT_SLACK[fact.hedged]
    per_row = []
    for r in rows:
        rng = ctx.alignment.to_reference(r)
        if rng is None:
            continue  # G3: unaligned device or unusable time
        can_conflict = allow_conflict and _can_conflict(r, ctx)
        per_row.append((_classify(rng, claim, m, c, can_conflict), r, rng))
    verdict, row = _combine(per_row)
    return verdict, row, (_time_quote(row) if row else None)


def judge_gap(fact: Fact, ctx: Context):
    """'<event> N minutes after <ref event>'."""
    if not (fact.event and fact.ref_event):
        return NOT_CHECKABLE, None, None
    ends = [r for r in ctx.rows if row_events(r) == {fact.event}]
    starts = [r for r in ctx.rows if row_events(r) == {fact.ref_event}]
    if not ends or not starts:
        return NOT_CHECKABLE, None, None
    if fact.value is None:
        return UNSURE, None, None  # "a few minutes": too vague to check
    want = fact.value * 60.0 * (1 if fact.direction == "after" else -1)
    m, c = GAP_MATCH_TOL[fact.hedged] * 60.0, GAP_CONFLICT_TOL[fact.hedged] * 60.0
    per_row = []
    for e in ends:
        for s in starts:
            if e.time is None or s.time is None:
                continue
            if e.device == s.device:
                # Same clock: its offset cancels out, so use the raw times.
                gap = (e.time.lo - s.time.hi, e.time.hi - s.time.lo)
            else:
                er, sr = ctx.alignment.to_reference(e), ctx.alignment.to_reference(s)
                if er is None or sr is None:
                    continue  # G3
                gap = (er[0] - sr[1], er[1] - sr[0])
            can_conflict = _can_conflict(e, ctx) and _can_conflict(s, ctx)
            per_row.append((_classify(gap, (want, want), m, c, can_conflict), e, gap))
    verdict, row = _combine(per_row)
    return verdict, row, (_time_quote(row) if row else None)


def _window_rows(fact: Fact, ctx: Context, field_of):
    """Rows that have the wanted field and, if the claim gives a time, lie
    entirely inside that time window on the reference clock."""
    rows = [r for r in ctx.rows if field_of(r)]
    if not rows:
        return None
    if fact.context is None or fact.context.clock is None:
        return rows, False
    claim = ctx.claim_range(fact.context.clock)
    if claim is None:
        return [], True
    m = MATCH_SLACK[fact.context.hedged]
    inside = []
    for r in rows:
        rng = ctx.alignment.to_reference(r)
        if rng is not None and _inside(rng, (claim[0] - m, claim[1] + m)):
            inside.append(r)
    return inside, True


def judge_speed(fact: Fact, ctx: Context):
    found = _window_rows(fact, ctx, _speed_field)
    if found is None:
        return NOT_CHECKABLE, None, None
    rows, timed = found
    if not timed:
        return UNSURE, None, None  # a speed with no time cannot be tied to a reading
    readings = [(r, _number(r.fields[_speed_field(r)])) for r in rows]
    readings = [(r, v) for r, v in readings if v is not None]
    tol = SPEED_MATCH_TOL[fact.hedged]
    # Without vehicle identity, only one clean reading in the window counts as support.
    if len(readings) == 1 and abs(readings[0][1] - fact.value) <= tol:
        r = readings[0][0]
        return MATCH, r, quote_for(r, [r.fields[_speed_field(r)]])
    return UNSURE, None, None


def judge_zone(fact: Fact, ctx: Context):
    found = _window_rows(fact, ctx, _zone_field)
    if found is None:
        return NOT_CHECKABLE, None, None
    rows, _ = found
    values = [(r, _number(r.fields[_zone_field(r)])) for r in rows]
    values = [(r, v) for r, v in values if v is not None]
    if values and all(v == fact.value for _, v in values):
        r = values[0][0]
        return MATCH, r, quote_for(r, [r.fields[_zone_field(r)]])
    return UNSURE, None, None  # a different posted limit could be another road: never conflict


def judge_count(fact: Fact, ctx: Context):
    # Counts are recognised but never decided: one cited row cannot prove "how many".
    return NOT_CHECKABLE, None, None


JUDGES = {"time": judge_time, "gap": judge_gap, "speed": judge_speed,
          "zone": judge_zone, "count": judge_count}


# ------------------------------------------------------------ whole claims

def judge_claim(claim: Claim, ctx: Context) -> dict:
    """A claim matches only if every fact in it matches. Any conflicting fact
    makes it a conflict. Otherwise not_settled, or not_checkable if nothing
    in it could be checked at all."""
    if not claim.facts:
        return make_finding(claim.text, NOT_CHECKABLE)
    results = [JUDGES[f.kind](f, ctx) for f in claim.facts]
    verdicts = [v for v, _, _ in results]
    if all(v == NOT_CHECKABLE for v in verdicts):
        return make_finding(claim.text, NOT_CHECKABLE)
    for wanted in (CONFLICT, MATCH):
        if wanted == CONFLICT and CONFLICT in verdicts:
            _, row, quote = next(r for r in results if r[0] == CONFLICT)
            return make_finding(claim.text, CONFLICT, row, quote, ctx.known)
        if wanted == MATCH and all(v == MATCH for v in verdicts):
            _, row, quote = results[0]
            return make_finding(claim.text, MATCH, row, quote, ctx.known)
    return make_finding(claim.text, UNSURE)


def judge_all(claims: List[Claim], ctx: Context) -> List[dict]:
    return [judge_claim(c, ctx) for c in claims]
