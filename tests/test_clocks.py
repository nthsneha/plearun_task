import os
import sys
import unittest
from datetime import date, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from verify.clocks import align, find_anchors  # noqa: E402
from verify.loaders import assign_times, infer_default_tz, load_records, load_text  # noqa: E402

EST = timezone(timedelta(hours=-5))
DAY = date(2026, 3, 2)


def rows_from(files):
    """files: {name: text}. Load and time-stamp them like the real pipeline does."""
    rows = []
    for name, text in files.items():
        rows.extend(load_text(name, text))
    assign_times(rows, EST, DAY)
    return rows


NOTES = "ID,Note,Time\nN-1,Caller reported weaving,21:11\nN-2,Stopped at lot,21:16\n"


class SampleClocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = load_records(os.path.join(ROOT, "records"))
        assign_times(rows, infer_default_tz(rows), DAY)
        cls.al = align(rows)
        cls.anchors = find_anchors(rows)

    def test_reference_is_the_best_connected_clock(self):
        self.assertEqual(self.al.reference, "notes_cp1252.csv")

    def test_dispatch_and_cameras_are_aligned_with_ranges(self):
        for device in ("dispatch.json", "CAM-03", "CAM-07"):
            self.assertTrue(self.al.is_aligned(device), device)
        lo, hi = self.al.offsets["dispatch.json"]
        self.assertTrue(lo <= 60 <= hi)        # notes minute "21:11" vs dispatch 21:10:19
        self.assertLessEqual(hi - lo, 300)

    def test_bodycam_is_left_unaligned(self):
        # "caller says weaving" (relayed, ~21:11) and "pulling in" (~21:16) are 2.5s apart in the
        # recording but five minutes apart on the reference clock: they cannot both be anchors.
        self.assertFalse(self.al.is_aligned("bodycam_transcript.vtt"))

    def test_duplicate_camera_row_counts_once(self):
        cam03_stops = [a for a in self.anchors if a.device == "CAM-03" and a.event == "stop"]
        self.assertEqual(len(cam03_stops), 1)


class SyntheticClocks(unittest.TestCase):
    def test_two_agreeing_anchors_align_tightly(self):
        rows = rows_from({
            "notes.csv": NOTES,
            "cad.csv": "time,type\n21:09:30,911_call\n21:14:31,unit_on_scene\n",
            "cad2.csv": "time,type\n21:10:00,911_call\n21:15:00,unit_on_scene\n",
        })
        al = align(rows)
        self.assertEqual(al.reference, "cad.csv")  # most shared events
        # cad2 is 30s behind cad at both the call and on-scene: offset near -30, and narrow.
        lo, hi = al.offsets["cad2.csv"]
        self.assertTrue(lo <= -30 <= hi)
        self.assertLess(hi - lo, 40)

    def test_contradicting_anchors_leave_device_unaligned(self):
        rows = rows_from({
            "notes.csv": NOTES,
            # call says "+90s", stop says "-200s": cannot both be true
            "dev.csv": "time,type\n21:09:30,911_call\n21:19:20,vehicle stopped\n",
        })
        al = align(rows)
        # Whichever becomes the reference, the other must not be aligned to it.
        self.assertFalse(al.is_aligned("dev.csv") and al.is_aligned("notes.csv"))
        other = "notes.csv" if al.reference == "dev.csv" else "dev.csv"
        self.assertIn("disagree", al.reasons[other])

    def test_wrong_timezone_is_not_treated_as_drift(self):
        rows = rows_from({
            "notes.csv": NOTES,
            "utc.json": '{"events": [{"type": "911_call", "logged": "2026-03-02T21:11:20Z"}]}',
        })
        al = align(rows)
        self.assertFalse(al.is_aligned("utc.json"))
        self.assertIn("implausibly large", al.reasons["utc.json"])

    def test_device_with_no_shared_events_is_unaligned(self):
        rows = rows_from({"notes.csv": NOTES, "cam.csv": "timestamp,device,speed_mph\n21:12:00,CAM-1,40\n"})
        self.assertFalse(align(rows).is_aligned("CAM-1"))

    def test_same_event_at_different_times_is_ambiguous(self):
        rows = rows_from({
            "notes.csv": NOTES,
            "cam.csv": "timestamp,device,speed_mph\n21:05:00,CAM-1,0\n21:15:40,CAM-1,0\n",
        })
        self.assertEqual([a for a in find_anchors(rows) if a.device == "CAM-1"], [])
        self.assertFalse(align(rows).is_aligned("CAM-1"))

    def test_to_reference_adds_the_offset_range(self):
        rows = rows_from({"notes.csv": NOTES, "cad.csv": "time,type\n21:10:00,911_call\n21:13:00,other\n"})
        al = align(rows)
        other = [r for r in rows if r.file == "cad.csv"][1]
        lo, hi = al.to_reference(other)
        self.assertLess(lo, hi)
        self.assertEqual(lo - other.time.lo, al.offsets["cad.csv"][0])


if __name__ == "__main__":
    unittest.main()
