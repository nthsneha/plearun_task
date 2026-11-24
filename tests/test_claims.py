import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from verify.claims import extract_claims, report_date, split_sentences  # noqa: E402
from verify.events import events_in, first_event  # noqa: E402


def facts(text):
    return [(f.kind, f.event, f.clock, f.value, f.hedged) for c in extract_claims(text) for f in c.facts]


class SampleReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "reports", "report_fictional_02.txt"), encoding="utf-8") as fh:
            cls.text = fh.read()
        cls.claims = extract_claims(cls.text)

    def test_claim_texts_are_copied_from_the_report(self):
        self.assertEqual([c.text for c in self.claims], [
            "Officer Lee first observed the blue sedan at about 21:14 near mile marker 4",
            "travelling 58 mph in a posted 45 zone",
            "The driver pulled into the Harbor Gas lot and stopped around 21:16",
            "A 911 call reporting the incident was logged at 21:11",
            "Dispatch records show a unit on scene roughly 2 minutes after the stop",
            "Officer Lee states the vehicle was the only car in the lane",
        ])
        flat = " ".join(self.text.split())
        for c in self.claims:
            self.assertIn(c.text, flat)

    def test_facts(self):
        c = self.claims
        self.assertEqual([(f.kind, f.event, f.clock, f.hedged) for f in c[0].facts],
                         [("time", "observe", "21:14", True)])
        self.assertEqual(sorted((f.kind, f.value) for f in c[1].facts), [("speed", 58.0), ("zone", 45.0)])
        self.assertEqual(c[1].facts[0].context.clock, "21:14")  # speed inherits the sentence's time
        self.assertEqual([(f.event, f.clock, f.hedged) for f in c[2].facts], [("stop", "21:16", True)])
        self.assertEqual([(f.kind, f.event, f.clock, f.hedged) for f in c[3].facts],
                         [("time", "call_911", "21:11", False)])
        gap = c[4].facts[0]
        self.assertEqual((gap.kind, gap.event, gap.ref_event, gap.value, gap.direction, gap.hedged),
                         ("gap", "on_scene", "stop", 2.0, "after", True))
        self.assertEqual(c[5].facts, [])  # nothing checkable -> not_checkable later

    def test_header_is_not_a_claim_and_gives_the_date(self):
        self.assertFalse(any("SYNTHETIC" in c.text or "Route 21" in c.text for c in self.claims))
        self.assertEqual(report_date(self.text), (2026, 3, 2))


class Phrasings(unittest.TestCase):
    def test_am_pm_and_seconds(self):
        self.assertEqual(facts("The car stopped at 9:16 pm."), [("time", "stop", "21:16", None, False)])
        self.assertEqual(facts("The car stopped at 21:16:30."), [("time", "stop", "21:16:30", None, False)])

    def test_nonsense_times_are_ignored(self):
        self.assertEqual(facts("The car stopped at 25:61."), [])

    def test_tilde_hedge(self):
        self.assertTrue(facts("Stopped at ~21:16.")[0][4])

    def test_seconds_gap_becomes_minutes(self):
        self.assertEqual(facts("A unit arrived 90 seconds after the stop.")[0][3], 1.5)

    def test_count(self):
        self.assertEqual(facts("Two units responded."), [("count", None, None, 2.0, False)])

    def test_sentence_with_decimals_and_abbreviations(self):
        self.assertEqual(len(split_sentences("Speed was 58.5 mph. It stopped.")), 2)


class Events(unittest.TestCase):
    def test_record_style_text(self):
        self.assertEqual(events_in("unit_on_scene"), {"on_scene"})
        self.assertEqual(events_in("Caller described vehicle weaving"), {"call_911"})
        self.assertEqual(events_in("Pulling in at the lot, about 21:16."), {"stop"})

    def test_negation(self):
        self.assertEqual(events_in("The driver did not stop"), set())
        self.assertEqual(events_in("never actually arrived"), set())
        self.assertEqual(first_event("did not stop, then pulled over"), "stop")


if __name__ == "__main__":
    unittest.main()
