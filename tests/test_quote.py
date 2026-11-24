import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from verify.loaders import load_records, load_text  # noqa: E402
from verify.quote import make_finding, quote_for  # noqa: E402


class QuoteFor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {(r.file, r.row): r for r in load_records(os.path.join(ROOT, "records"))}

    def test_csv_quote_is_the_line(self):
        row = self.rows[("camera_log.csv", 13)]
        self.assertEqual(quote_for(row, ["58"]), "1772503638000,CAM-07,58,45")

    def test_json_quote_is_just_the_needed_lines(self):
        row = self.rows[("dispatch.json", 1)]
        self.assertEqual(quote_for(row, ["2026-03-02 21:10:19"]), '"logged": "2026-03-02 21:10:19"')
        q = quote_for(row, ["911_call", "2026-03-02 21:10:19"])
        self.assertTrue(q.startswith('"type": "911_call"') and q.endswith('"2026-03-02 21:10:19"'))
        self.assertIn(q, row.raw)

    def test_value_not_written_verbatim_falls_back_to_whole_row(self):
        row = load_text("n.csv", 'ID,Note,Time\nN-1,"said ""stop""",21:15\n')[0]
        self.assertEqual(quote_for(row, ['said "stop"']), row.raw)


class MakeFinding(unittest.TestCase):
    def setUp(self):
        self.row = load_text("notes.csv", "ID,Note,Time\nN-3,Stopped at lot ~21:16,21:16\n")[0]

    def test_good_match_keeps_cite_and_quote(self):
        f = make_finding("stopped around 21:16", "match", self.row, "Stopped at lot ~21:16")
        self.assertEqual(f, {"claim": "stopped around 21:16", "finding": "match",
                             "cite": {"file": "notes.csv", "row": 1}, "quote": "Stopped at lot ~21:16"})

    def test_quote_not_in_row_is_downgraded(self):
        f = make_finding("c", "conflict", self.row, "Stopped at 21:20")
        self.assertEqual(f, {"claim": "c", "finding": "not_settled"})

    def test_empty_or_missing_quote_or_row_is_downgraded(self):
        for row, q in [(self.row, ""), (self.row, "   "), (self.row, None), (None, "Stopped")]:
            with self.subTest(q=q):
                self.assertEqual(make_finding("c", "match", row, q)["finding"], "not_settled")

    def test_cite_of_a_row_that_does_not_exist_is_downgraded(self):
        f = make_finding("c", "match", self.row, "Stopped", known_rows={("notes.csv", 2)})
        self.assertEqual(f["finding"], "not_settled")

    def test_non_decisive_findings_never_carry_cite_or_quote(self):
        for v in ("not_settled", "not_checkable"):
            self.assertEqual(make_finding("c", v, self.row, "Stopped"), {"claim": "c", "finding": v})

    def test_unknown_verdict_is_an_error(self):
        with self.assertRaises(ValueError):
            make_finding("c", "probably", self.row, "Stopped")


if __name__ == "__main__":
    unittest.main()
