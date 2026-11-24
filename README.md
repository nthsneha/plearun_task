# Verified Findings

`python3 plearun_cli.py REPORT RECORDS_DIR > findings.json` (`--explain` shows clock alignment). `make test` runs 121 offline tests. Python 3.9+, standard library only, deterministic.

## How it works
1. **Load** (`loaders.py`): detects each file's format from its content. Each row keeps its exact raw text and is numbered from 1 (CSV data row, JSON event, VTT cue).
2. **Times** (`timeparse.py`): every timestamp becomes a range sized to its precision (`21:11` = 60 s).
3. **Claims** (`claims.py`): the report is split into verbatim claims with typed facts: time, speed, zone, gap, count.
4. **Clocks** (`clocks.py`): events seen by several devices give each device an offset *range* against a reference clock. Only record rows are used, never the report.
5. **Judge** (`judge.py`): match if the evidence fits a small window across the whole range, conflict if it misses a much larger one, otherwise not_settled.
6. **Gate** (`quote.py`): a finding whose cite is not a real row, or whose quote is not an exact substring of that row, becomes not_settled.

Optional: `--llm mock|clef` adds a separate LLM pass afterwards (`llm_step.py`, see below).

## Three hardest decisions
1. **Ranges, not best guesses.** The tool decides only when the answer holds for every plausible clock offset. So "on scene roughly 2 minutes after the stop" is not_settled, because the corrected gap could be anywhere from 0.7 to 3.3 min.
2. **When to leave a device unaligned.** The body cam's "caller says weaving" is a relay, not the call. Its two anchors imply start times five minutes apart, so it stays unaligned instead of being forced to fit. An offset of hours is treated as a timezone error, not drift.
3. **Conflicts harder to reach than matches.** The conflict window is much wider than the match window, and weak records can never cause a conflict on their own (see below).

## What it gets wrong
- It assumes the report's times are on the reference clock. Bare `HH:MM` times break across midnight.
- Claim extraction uses regular expressions, so phrasings it doesn't know become not_checkable. Counts are never decided, and speeds rarely are.
- Combining ranges ignores the correlation between offsets, so chained comparisons come out wider than they need to be.

## What would break first on real records
The event vocabulary in `events.py` (CAD codes like `10-97`), then incidents with several vehicles, since nothing ties a camera reading to this car.

## Tools used, and the calls I made
I used **Claude Code (Claude Opus 5.5)** to write the code. I talked the design through with it, then had it build the tool in nine small steps, approving each one after it was explained and tested. I used `unittest` for the tests and `cProfile` for profiling.

Decisions I made beyond the three above:
- **One flat list of rows.** Every file, nested JSON included, is read once into one list that claims are checked against. It's simple and dependency-free, and handles 15 MB in ~10 s because each row's event is computed once (profiling showed repeated work).
- **A rule-based core.** Fixed rules mean the same input always gives the same output. A test confirms this byte for byte across three Python hash seeds.
- **A vague record doesn't cancel a clear one.** A clear match stands even if another record is only roughly placed in time (wide clock offset). A record pointing clearly elsewhere makes it not_settled.
- **Weak records can confirm a conflict, never start one.** A camera reading (no vehicle ID) or a guessed-timezone time can't cause a conflict alone, but shouldn't cancel one a reliable record shows. Found by testing: moving the sample's 911 time to 21:30 gave not_settled, though both records put the call ~20 min earlier.
- **The LLM is a separate add-on.** It runs only with `--llm`, after the normal pipeline, on claims no rule could check. The core never imports it.
- **Why Clef.** Rule 6 lets an LLM only choose among candidate rows. Clef is a decision model: it returns a probability per option from a fixed set instead of writing text. I ask one `choice` question over the candidate row IDs plus "none", so it can't invent a row, and keep picks at ≥90% probability, which the same rules then re-check. Tested it out with the live API.
