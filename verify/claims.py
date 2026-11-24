"""Split the report into claims, and pull the checkable facts out of each one.

A claim is a piece of the report copied word for word. Usually it is one
sentence. When a sentence states facts in separate comma-clauses
("observed ... at about 21:14, travelling 58 mph ..."), it is split there,
so each claim can get its own finding.

Each claim holds zero or more facts:
  time   an event happened at a clock time       ("stopped around 21:16")
  speed  the vehicle's speed                      ("58 mph")
  zone   the posted speed limit                   ("a posted 45 zone")
  gap    time between two events                  ("on scene roughly 2 minutes after the stop")
  count  how many of something                    ("two units")
A claim with no facts gets a not_checkable finding.

Hedge words (about, around, roughly, ~ ...) are recorded on each fact, so
the judge can allow more slack for them.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from verify.events import first_event, last_event_before

HEDGE_RE = re.compile(
    r"(?:\babout|\baround|\broughly|\bapproximately|\bapprox\.?|\bcirca|~|\bnearly|\balmost"
    r"|\bjust (?:after|before)|\bshortly (?:after|before)|\bsome)\s*$", re.I)

TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([ap])\.?m\.?\b)?(?![\d:])", re.I)
ZONE_RE = re.compile(
    r"\bposted\s+(?:speed\s+)?(?:limit\s+(?:of\s+)?)?(\d{1,3})(?:\s*mph)?(?:\s+(?:mph\s+)?zone)?"
    r"|\b(\d{1,3})(?:\s*mph)?\s+zone\b"
    r"|\bspeed limit (?:of |was |is )?(\d{1,3})", re.I)
SPEED_RE = re.compile(r"\b(\d{1,3})\s*(?:mph|miles per hour|mi/h)\b|\b(\d{1,3}) in a \d{1,3}\b", re.I)
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2}
_NUM = r"(\d+(?:\.\d+)?|a couple of|an?|one|two|three|four|five|six|seven|eight|nine|ten)"
GAP_RE = re.compile(_NUM + r"\s+(seconds?|secs?|minutes?|mins?)\s+(after|before|later than|prior to)\s+"
                    r"(?:the\s+)?([^,.;]{1,40})", re.I)
# (?!911\b): "a 911 call" names the emergency line, it is not 911 calls.
COUNT_RE = re.compile(r"\b(?!911\b)" + _NUM + r"\s+(units?|officers?|vehicles?|cars?|calls?|occupants?|passengers?)\b", re.I)
LABEL_RE = re.compile(r"\bSYNTHETIC\b|\bFICTIONAL\b|^Incident\s+\d{4}-\d{2}-\d{2}\b", re.I)
DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass
class Fact:
    kind: str                      # "time" | "speed" | "zone" | "gap" | "count"
    hedged: bool = False
    event: Optional[str] = None    # time/gap: the event the claim is about
    clock: Optional[str] = None    # time: "21:14" as written (converted to 24h)
    value: Optional[float] = None  # speed/zone mph, gap minutes, count number
    ref_event: Optional[str] = None  # gap: the event it is measured from
    direction: Optional[str] = None  # gap: "after" or "before"
    noun: Optional[str] = None     # count: what is counted
    context: Optional["Fact"] = None  # speed/zone: the time fact of the same sentence, if any


@dataclass
class Claim:
    text: str
    facts: List[Fact] = field(default_factory=list)


def report_date(text: str):
    """First ISO date written in the report (e.g. 'Incident 2026-03-02'), as (y, m, d), or None."""
    m = DATE_RE.search(text)
    return tuple(int(g) for g in m.groups()) if m else None


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.replace("﻿", "")).strip()
    # A sentence ends at . ! or ? followed by a space and a capital letter or digit.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _hedged(sentence: str, start: int) -> bool:
    return bool(HEDGE_RE.search(sentence[:start]))


def _facts_in(chunk: str) -> List[Fact]:
    """Find every fact in one chunk of text."""
    facts: List[Fact] = []
    taken = []  # character ranges already used, so "45 zone" is not also read as a speed

    for m in GAP_RE.finditer(chunk):
        n, unit, direction, ref_text = m.groups()
        word = n.lower()
        value = NUMBER_WORDS.get(word, None) if not word[0].isdigit() else float(n)
        if value is not None and unit.lower().startswith("sec"):
            value = value / 60.0
        facts.append(Fact("gap", hedged=_hedged(chunk, m.start()) or word == "a couple of",
                          event=first_event(chunk[:m.start()]), value=value,
                          ref_event=first_event(ref_text),
                          direction="before" if direction.lower() in ("before", "prior to") else "after"))
        taken.append(m.span())

    for m in ZONE_RE.finditer(chunk):
        facts.append(Fact("zone", value=float(next(g for g in m.groups() if g))))
        taken.append(m.span())

    for m in SPEED_RE.finditer(chunk):
        if any(s <= m.start() < e for s, e in taken):
            continue
        facts.append(Fact("speed", hedged=_hedged(chunk, m.start()), value=float(m.group(1) or m.group(2))))
        taken.append(m.span())

    for m in TIME_RE.finditer(chunk):
        hour, minute, sec, ampm = m.groups()
        h = int(hour)
        if ampm:
            if h > 12:
                continue  # "21:00 pm" is nonsense; do not guess
            h = h % 12 + (12 if ampm.lower() == "p" else 0)
        if h > 23 or int(minute) > 59:
            continue
        clock = f"{h:02d}:{minute}" + (f":{sec}" if sec else "")
        # The event is the one mentioned closest before the time, else anywhere in the chunk.
        event = last_event_before(chunk, m.start()) or first_event(chunk)
        facts.append(Fact("time", hedged=_hedged(chunk, m.start()), event=event, clock=clock))
        taken.append(m.span())

    for m in COUNT_RE.finditer(chunk):
        if any(s <= m.start() < e for s, e in taken):
            continue
        word = m.group(1).lower()
        value = float(word) if word[0].isdigit() else NUMBER_WORDS.get(word)
        if word in ("a", "an"):
            continue  # "a unit" is not a count claim
        facts.append(Fact("count", value=value, noun=m.group(2).lower().rstrip("s")))
    return facts


def _chunks(sentence: str) -> List[str]:
    """Split a sentence at commas only where a new comma-clause starts a new fact
    and the text so far already has one. Clauses without facts stay attached."""
    clauses = [c for c in re.split(r",\s+", sentence)]
    chunks: List[str] = []
    for clause in clauses:
        if chunks and not (_facts_in(clause) and _facts_in(chunks[-1])):
            chunks[-1] = chunks[-1] + ", " + clause
        else:
            chunks.append(clause)
    return chunks


def extract_claims(report_text: str) -> List[Claim]:
    claims: List[Claim] = []
    for sentence in split_sentences(report_text):
        if LABEL_RE.search(sentence):
            continue  # report header / labels, not claims
        sentence = sentence.rstrip(".!?").strip()
        chunk_claims = [Claim(text=c.strip(), facts=_facts_in(c)) for c in _chunks(sentence)]
        # Speed and zone facts inherit the time of the same sentence ("... at 21:14, travelling 58 mph").
        times = [f for c in chunk_claims for f in c.facts if f.kind == "time"]
        for c in chunk_claims:
            for f in c.facts:
                if f.kind in ("speed", "zone") and len(times) == 1:
                    f.context = times[0]
        claims.extend(chunk_claims)
    return claims
