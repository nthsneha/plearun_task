"""Turn the timestamp text found in records and reports into time ranges.

Every time becomes a Stamp: a range [lo, hi] in seconds. The width of the
range says how precise the original text was. "21:11" could be any second
from 21:11:00 to 21:11:59, so it becomes a 60-second range. An epoch in
milliseconds is exact, so lo == hi.

Absolute stamps are counted in seconds since 1970-01-01 UTC, so times from
different files can be compared directly. Relative stamps (body-cam cue
times like 00:00:03.500) count seconds since the recording started. They
have no clock time until the device is aligned.

Anything that does not look exactly like a known format returns None.
Callers treat that as "this row has no usable time" and never guess.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union

# Epochs outside 2000-01-01 .. 2100-01-01 are treated as garbage, not times.
_EPOCH_MIN = 946684800
_EPOCH_MAX = 4102444800

_EPOCH_RE = re.compile(r"(\d{13})|(\d{10})(\.\d{1,6})?")
_ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(\.\d{1,6})?)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?"
)
_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
_OFFSET_RE = re.compile(r"([+-])(\d{2}):?(\d{2})")
_CUE_RE = re.compile(r"(?:(\d{1,3}):)?(\d{2}):(\d{2})\.(\d{3})")


@dataclass(frozen=True)
class Stamp:
    lo: float                 # earliest second this time could mean
    hi: float                 # latest second this time could mean
    relative: bool = False    # True for recording-relative cue times
    assumed_tz: bool = False  # True when the text had no timezone and we supplied one

    @property
    def width(self) -> float:
        return self.hi - self.lo


def parse_offset(text: str) -> Optional[timezone]:
    """'-05:00' or '-0500' -> a timezone object. 'Z' -> UTC. Anything else -> None."""
    text = text.strip()
    if text == "Z":
        return timezone.utc
    m = _OFFSET_RE.fullmatch(text)
    if not m:
        return None
    sign, hours, minutes = m.groups()
    if int(hours) > 23 or int(minutes) > 59:
        return None
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    return timezone(-delta if sign == "-" else delta)


def _clean(value: Union[str, int, float, None]) -> Optional[str]:
    """Normalise the input to stripped text. Booleans are refused: True is an int in Python."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    return value.strip("﻿ \t\r\n")


def parse_timestamp(value, default_tz: timezone, base_date: date) -> Optional[Stamp]:
    """Parse an absolute time from a record field or the report.

    default_tz is used when the text has no timezone (the result is flagged
    assumed_tz). base_date is used for bare clock times like "21:15".
    """
    text = _clean(value)
    if not text:
        return None

    m = _EPOCH_RE.fullmatch(text)
    if m:
        millis, secs, frac = m.groups()
        if millis:
            lo, width = int(millis) / 1000.0, 0.0
        else:
            lo = int(secs) + (float(frac) if frac else 0.0)
            width = 10.0 ** -(len(frac) - 1) if frac else 1.0
        if not _EPOCH_MIN <= lo <= _EPOCH_MAX:
            return None
        return Stamp(lo, lo + width)

    m = _ISO_RE.fullmatch(text)
    if m:
        year, month, day, hour, minute, sec, frac, tz_text = m.groups()
        tz = parse_offset(tz_text) if tz_text else default_tz
        if tz is None:
            return None
        try:
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute),
                          int(sec or 0), tzinfo=tz)
        except ValueError:  # e.g. month 13 or hour 25
            return None
        lo = dt.timestamp() + (float(frac) if frac else 0.0)
        if frac:
            width = 10.0 ** -(len(frac) - 1)
        elif sec is not None:
            width = 1.0
        else:
            width = 60.0
        return Stamp(lo, lo + width, assumed_tz=tz_text is None)

    m = _CLOCK_RE.fullmatch(text)
    if m:
        hour, minute, sec = m.groups()
        try:
            dt = datetime(base_date.year, base_date.month, base_date.day,
                          int(hour), int(minute), int(sec or 0), tzinfo=default_tz)
        except ValueError:
            return None
        lo = dt.timestamp()
        return Stamp(lo, lo + (1.0 if sec is not None else 60.0), assumed_tz=True)

    return None


def parse_cue_time(value) -> Optional[Stamp]:
    """Parse a WebVTT cue time ('00:00:03.500' or '00:03.500') as seconds since the recording started."""
    text = _clean(value)
    if not text:
        return None
    m = _CUE_RE.fullmatch(text)
    if not m:
        return None
    hours, minutes, secs, millis = m.groups()
    if int(minutes) > 59 or int(secs) > 59:
        return None
    t = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000.0
    return Stamp(t, t, relative=True)
