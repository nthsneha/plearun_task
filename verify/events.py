"""Recognise which real-world event a piece of text talks about.

Used on both sides: report claims ("stopped around 21:16") and record rows
("Stopped at lot ~21:16", a dispatch type of "unit_on_scene"). Sharing one
list keeps the two sides speaking the same language.

A mention right after a negation ("did not stop", "never arrived") does not count.
"""
import re
from typing import List, Optional, Set, Tuple

EVENT_PATTERNS = {
    "call_911":   r"\b911\b|\bcaller\b|\bcall(?:ed)? in\b",
    "stop":       r"\bstop(?:ped|s)?\b|\bpull(?:ed|ing|s)? (?:in|into|over)\b|\bparked\b",
    "observe":    r"\bobserv(?:ed|es|ing)\b|\bspotted\b|\bfirst saw\b",
    "on_scene":   r"\bon[ -]?scene\b|\barriv(?:ed|es|al)\b",
    "dispatched": r"\bdispatched\b|\ben ?route\b",
}
_COMPILED = {name: re.compile(p, re.I) for name, p in EVENT_PATTERNS.items()}
# A negation word, then at most one other word, right before the mention.
_NEGATION = re.compile(r"\b(?:not|never|no|didn't|wasn't|hadn't|without)\W+(?:\w+\W+)?$", re.I)


def mentions(text: str) -> List[Tuple[int, str]]:
    """Every non-negated event mention as (position, event name), in text order."""
    text = text.replace("_", " ")  # "unit_on_scene" -> "unit on scene", so field values read like prose
    found = []
    for name, rx in _COMPILED.items():
        for m in rx.finditer(text):
            if not _NEGATION.search(text[:m.start()]):
                found.append((m.start(), name))
    return sorted(found)


def events_in(text: str) -> Set[str]:
    return {name for _, name in mentions(text)}


def first_event(text: str) -> Optional[str]:
    found = mentions(text)
    return found[0][1] if found else None


def last_event_before(text: str, pos: int) -> Optional[str]:
    """The event mentioned last before position pos: in 'stopped around 21:16' that is 'stop'."""
    found = mentions(text[:pos])
    return found[-1][1] if found else None
