"""Build findings, and enforce rule 1 at the last moment.

make_finding() is the only place a finding dict is created. A "match" or
"conflict" is written only if:
  - it cites a real row (one the loaders produced from a real file), and
  - its quote is a non-empty, exact substring of that row's raw text.
If either check fails, the finding is downgraded to "not_settled" and the
cite and quote are dropped. No other code can skip this check.
"""
from typing import Iterable, Optional

from verify.rows import Row

VERDICTS = ("match", "conflict", "not_settled", "not_checkable")
DECISIVE = ("match", "conflict")


def quote_for(row: Row, values: Iterable[str]) -> str:
    """Pick a readable quote from the row's raw text that contains the given values.

    Takes the whole line(s) of raw text that cover every value, e.g. one
    CSV line, or the '"logged": ...' line of a JSON event. The result is cut
    straight out of row.raw, so it is a substring by construction. If any
    value is not written verbatim in the raw text (CSV doubled quotes, for
    example), the whole raw row is used instead.
    """
    raw = row.raw
    starts, ends = [], []
    for v in values:
        v = str(v)
        pos = raw.find(v) if v else -1
        if pos < 0:
            return raw.strip()
        starts.append(pos)
        ends.append(pos + len(v))
    if not starts:
        return raw.strip()
    lo = raw.rfind("\n", 0, min(starts)) + 1          # start of the first line used
    hi = raw.find("\n", max(ends))                    # end of the last line used
    return raw[lo:hi if hi >= 0 else len(raw)].strip()


def quote_is_exact(row: Optional[Row], quote: Optional[str]) -> bool:
    return (isinstance(row, Row) and isinstance(quote, str) and quote.strip() != ""
            and quote in row.raw)


def make_finding(claim_text: str, verdict: str, row: Optional[Row] = None,
                 quote: Optional[str] = None, known_rows: Optional[set] = None) -> dict:
    """The output dict for one claim, in the exact required shape.

    known_rows, if given, is the set of (file, row) pairs that really exist;
    a cite outside it is treated as fake.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    if verdict in DECISIVE:
        real = row is not None and (known_rows is None or (row.file, row.row) in known_rows)
        if not (real and quote_is_exact(row, quote)):
            verdict = "not_settled"
    finding = {"claim": claim_text, "finding": verdict}
    if verdict in DECISIVE:
        finding["cite"] = {"file": row.file, "row": row.row}
        finding["quote"] = quote
    return finding
