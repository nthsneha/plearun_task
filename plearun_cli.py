"""Check each claim in an incident report against machine records.

Usage: python3 plearun_cli.py REPORT RECORDS_DIR [--explain] [--llm mock|clef] > findings.json

By default it works fully offline with rules only, no LLM. --explain prints
how each clock was aligned to stderr; stdout stays pure JSON.

--llm adds an optional extra pass AFTER the normal pipeline has finished
(verify/llm_step.py). The core steps never load it. "clef" needs network
access and Cloudflare credentials.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone

from verify.claims import extract_claims, report_date
from verify.clocks import align
from verify.judge import Context, judge_all
from verify.loaders import assign_times, decode_bytes, infer_default_tz, load_records


def _fallback_date(rows) -> date:
    """If the report states no date, use the most common date among rows with a full date."""
    counts = Counter()
    for r in rows:
        if r.time is not None and not r.time.relative and not r.time.assumed_tz:
            counts[datetime.fromtimestamp(r.time.lo, timezone.utc).date()] += 1
    return counts.most_common(1)[0][0] if counts else date(1970, 1, 1)


def _load_provider(name):
    """Imported only on request, so the default run never loads LLM or network code."""
    if name == "mock":
        import mock_llm
        return mock_llm.complete
    from providers import clef
    return clef.complete


def run(report_path: str, records_dir: str, explain: bool = False, llm: str = None) -> dict:
    with open(report_path, "rb") as fh:
        report = decode_bytes(fh.read())
    claims = extract_claims(report)

    rows = load_records(records_dir)
    tz = infer_default_tz(rows) or timezone.utc
    ymd = report_date(report)
    if ymd:
        base = date(*ymd)
        assign_times(rows, tz, base)
    else:
        assign_times(rows, tz, date(1970, 1, 1))   # first pass: only full-date rows matter
        base = _fallback_date(rows)
        assign_times(rows, tz, base)

    alignment = align(rows)
    if explain:
        print(f"reference clock: {alignment.reference}", file=sys.stderr)
        for device in sorted(alignment.reasons):
            off = alignment.offsets.get(device)
            shown = f" offset {off[0]:+.0f}s..{off[1]:+.0f}s" if off else ""
            print(f"  {device}: {alignment.reasons[device]}{shown}", file=sys.stderr)

    ctx = Context(rows, alignment, tz, base)
    findings = judge_all(claims, ctx)          # the core tool ends here

    if llm:                                    # optional add-on, only on request
        from verify.llm_step import run_llm_step
        findings = run_llm_step(claims, findings, ctx, _load_provider(llm))
    return {"findings": findings}


def main(argv) -> int:
    parser = argparse.ArgumentParser(prog="plearun_cli.py", description="Check report claims against records.")
    parser.add_argument("report")
    parser.add_argument("records_dir")
    parser.add_argument("--explain", action="store_true", help="print clock alignment to stderr")
    parser.add_argument("--llm", choices=["mock", "clef"], help="optional extra LLM pass after the normal run (off by default)")
    args = parser.parse_args(argv)   # bad arguments: usage on stderr, exit code 2
    if not os.path.isfile(args.report) or not os.path.isdir(args.records_dir):
        parser.error("need a report file and a records folder")
    result = run(args.report, args.records_dir, explain=args.explain, llm=args.llm)
    # ensure_ascii keeps stdout plain ASCII, so any console encoding can write it.
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
