"""Read every record file into a flat list of Rows.

Each file's format is detected from its content, not its name:
  starts with "WEBVTT"       -> body-cam style transcript
  starts with "{" or "["     -> JSON export
  anything else              -> CSV

A file that cannot be read is reported on stderr and skipped. One bad file
must never stop the whole run.

Record text is only ever stored as data. Nothing in it is run or obeyed.
"""
import csv
import io
import json
import os
import re
import sys
from collections import Counter
from datetime import date, timezone
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple

from verify.rows import Row
from verify.timeparse import parse_cue_time, parse_offset, parse_timestamp

# Column/key names that usually hold the row's time, checked before other fields.
_TIME_NAME_RE = re.compile(r"time|timestamp|logged|^ts$|date|^at$|when", re.I)
_TRAILING_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
_WORDY_RE = re.compile(r"[A-Za-z]{3}")        # a value with a real word in it, not just digits/codes
# The only characters that matter for brace matching: an escape pair, a quote, or a brace.
_JSON_TOKEN_RE = re.compile(r'\\.|["{}]', re.S)


def decode_bytes(data: bytes) -> str:
    """UTF-8 (with or without BOM) first. If that fails, Windows-1252 (cp1252).
    cp1252 leaves five byte values undefined. Those become U+FFFD instead of crashing."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def load_records(records_dir: str) -> List[Row]:
    """Load every regular, non-hidden file in records_dir, in name order."""
    rows: List[Row] = []
    for name in sorted(os.listdir(records_dir)):
        path = os.path.join(records_dir, name)
        if name.startswith(".") or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                text = decode_bytes(fh.read())
            rows.extend(load_text(name, text))
        except (ValueError, RecursionError, csv.Error) as exc:
            print(f"warning: skipped {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return rows


def load_text(name: str, text: str) -> List[Row]:
    head = text.lstrip()
    if head.startswith("WEBVTT"):
        return list(_load_vtt(name, text))
    if head and head[0] in "{[":
        return list(_load_json(name, text))
    return list(_load_csv(name, text))


# ---------------------------------------------------------------- CSV

def _load_csv(name: str, text: str) -> Iterator[Row]:
    lines = text.splitlines()  # also removes the \r of Windows line endings
    # Skip marker/comment lines ("#PLEARUN ...") and blank lines before the header.
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    body = lines[start:]
    if not body:
        return
    reader = csv.reader(io.StringIO("\n".join(body)))
    header = [h.strip() for h in next(reader)]
    data_row = 0
    prev_line = reader.line_num  # physical lines used so far (the header)
    for values in reader:
        # One CSV record can span several physical lines when a quoted field
        # contains a line break. line_num tells us where this record ended.
        raw = "\n".join(body[prev_line:reader.line_num])
        prev_line = reader.line_num
        if not any(v.strip() for v in values):
            continue  # blank line: not a data row
        data_row += 1
        fields = {h: v for h, v in zip(header, values)}
        yield _make_row(name, data_row, raw, "csv", fields)


# ---------------------------------------------------------------- JSON

def _object_spans(text: str) -> List[Tuple[int, int]]:
    """Start/end position of every {...} in the text, in the order the braces open.
    String-aware, so braces inside quoted text are ignored."""
    spans, stack = [], []
    in_string = False
    # Jump between interesting characters instead of looking at every one.
    for m in _JSON_TOKEN_RE.finditer(text):
        tok, i = m.group(), m.start()
        if tok[0] == "\\":
            continue  # an escaped character such as \" inside a string: never a real quote
        if tok == '"':
            in_string = not in_string
        elif in_string:
            continue  # a brace inside quoted text
        elif tok == "{":
            spans.append([i, None])
            stack.append(len(spans) - 1)
        elif stack:  # "}"
            spans[stack.pop()][1] = i + 1
    return [(s, e) for s, e in spans]


def _load_json(name: str, text: str) -> Iterator[Row]:
    data = json.loads(text)
    spans = _object_spans(text)
    # Walk the parsed data in document order with an explicit stack (no recursion,
    # so deeply nested files cannot overflow Python's call stack). The n-th object
    # met in this walk is the n-th "{" in the text, which gives us its raw text.
    stack = [(data, False)]
    obj_index = -1
    event_no = 0
    while stack:
        node, in_list = stack.pop()
        if isinstance(node, dict):
            obj_index += 1
            # A row is any object that sits inside a list, e.g. export.events[*].
            if in_list:
                fields = _flatten_scalars(node)
                if fields:
                    event_no += 1
                    s, e = spans[obj_index] if obj_index < len(spans) else (0, 0)
                    yield _make_row(name, event_no, text[s:e], "json", fields)
            children = [(v, False) for v in node.values()]
        elif isinstance(node, list):
            children = [(v, True) for v in node]
        else:
            continue
        stack.extend(reversed(children))  # reversed so the first child is visited first


def _flatten_scalars(obj: dict) -> dict:
    """Keep this object's own text/number values, plus values of nested objects as
    'parent.child' keys. Lists are left alone: their objects become rows of their own."""
    out, todo = {}, [("", obj)]
    while todo:
        prefix, cur = todo.pop()
        for k, v in cur.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                todo.append((key + ".", v))
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
                out[key] = str(v)
    return out


# ---------------------------------------------------------------- WebVTT

def _load_vtt(name: str, text: str) -> Iterator[Row]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    cue_no = 0
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        timing_at = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_at is None or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        start_text, _, rest = lines[timing_at].partition("-->")
        end_text = rest.split()[0] if rest.split() else ""
        start, end = parse_cue_time(start_text), parse_cue_time(end_text)
        if start is None:
            continue
        cue_no += 1
        words = "\n".join(lines[timing_at + 1:])
        row = Row(file=name, row=cue_no, raw=words, fmt="vtt", device=name,
                  fields={"start": start_text.strip(), "end": end_text.strip()},
                  text=words, time_text=start_text.strip(), time=start)
        if end is not None:
            row.fields["end_seconds"] = str(end.lo)
        yield row


# ---------------------------------------------------------------- shared

@lru_cache(maxsize=None)
def _is_time_key(key: str) -> bool:
    return bool(_TIME_NAME_RE.search(key))


def _make_row(name: str, number: int, raw: str, fmt: str, fields: dict) -> Row:
    device = next((v for k, v in fields.items() if k.lower() == "device" and v.strip()), name)
    time_key = next((k for k in fields if _is_time_key(k)), None)
    text = " ".join(v for k, v in fields.items() if k != time_key and _WORDY_RE.search(v))
    return Row(file=name, row=number, raw=raw, fmt=fmt, device=device.strip(),
               fields=fields, text=text, time_text=fields.get(time_key) if time_key else None)


def infer_default_tz(rows: List[Row]) -> Optional[timezone]:
    """The most common explicit UTC offset written in the records (e.g. -05:00).
    Used for times written without one. None if no record states an offset."""
    counts = Counter()
    for r in rows:
        m = _TRAILING_OFFSET_RE.search((r.time_text or "").strip())
        if m and m.group(1) != "Z":
            tz = parse_offset(m.group(1))
            if tz is not None:
                counts[tz] += 1
    return counts.most_common(1)[0][0] if counts else None


def assign_times(rows: List[Row], default_tz: timezone, base_date: date) -> None:
    """Parse each row's time_text into a Stamp. Rows whose time cannot be parsed keep time=None."""
    for r in rows:
        if r.fmt != "vtt" and r.time_text is not None:
            r.time = parse_timestamp(r.time_text, default_tz, base_date)
