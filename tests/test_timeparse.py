import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify.timeparse import parse_cue_time, parse_offset, parse_timestamp  # noqa: E402

EST = timezone(timedelta(hours=-5))
DAY = date(2026, 3, 2)


def local(h, m, s=0):
    """Seconds since 1970 for a clock time on the sample day at -05:00."""
    return datetime(2026, 3, 2, h, m, s, tzinfo=EST).timestamp()


def p(value):
    return parse_timestamp(value, EST, DAY)


class EpochTimes(unittest.TestCase):
    def test_epoch_millis_is_exact(self):
        st = p("1772502579000")
        self.assertEqual((st.lo, st.hi), (local(20, 49, 39), local(20, 49, 39)))
        self.assertFalse(st.assumed_tz)

    def test_epoch_millis_as_json_int(self):
        self.assertEqual(p(1772504040000).lo, local(21, 14, 0))

    def test_epoch_seconds_is_one_second_wide(self):
        st = p("1772504040")
        self.assertEqual((st.lo, st.width), (local(21, 14, 0), 1.0))

    def test_out_of_range_epoch_is_rejected(self):
        self.assertIsNone(p("9999999999999"))
        self.assertIsNone(p("0000000000"))


class IsoTimes(unittest.TestCase):
    def test_iso_with_offset(self):
        st = p("2026-03-02T20:51:06-05:00")
        self.assertEqual((st.lo, st.width), (local(20, 51, 6), 1.0))
        self.assertFalse(st.assumed_tz)

    def test_iso_z_means_utc(self):
        self.assertEqual(p("2026-03-03T02:10:19Z").lo, local(21, 10, 19))

    def test_naive_datetime_uses_default_tz_and_says_so(self):
        st = p("2026-03-02 21:10:19")
        self.assertEqual(st.lo, local(21, 10, 19))
        self.assertTrue(st.assumed_tz)

    def test_fractional_seconds(self):
        st = p("2026-03-02T21:10:19.250-05:00")
        self.assertAlmostEqual(st.lo, local(21, 10, 19) + 0.25)
        # Large float times (~1.77e9) can only hold about 6 decimal places.
        self.assertAlmostEqual(st.width, 0.001, delta=1e-6)

    def test_impossible_dates_are_rejected(self):
        self.assertIsNone(p("2026-13-02T21:10:19-05:00"))
        self.assertIsNone(p("2026-03-02T25:10:19-05:00"))
        self.assertIsNone(p("2026-03-02T21:10:19-99:00"))


class ClockTimes(unittest.TestCase):
    def test_hh_mm_is_a_whole_minute(self):
        st = p("21:15")
        self.assertEqual((st.lo, st.width), (local(21, 15), 60.0))
        self.assertTrue(st.assumed_tz)

    def test_hh_mm_ss(self):
        self.assertEqual(p("21:15:30").width, 1.0)

    def test_surrounding_whitespace_and_bom_are_ignored(self):
        self.assertEqual(p("﻿ 21:15\r\n").lo, local(21, 15))


class CueTimes(unittest.TestCase):
    def test_cue_time_is_relative(self):
        st = parse_cue_time("00:00:03.500")
        self.assertEqual((st.lo, st.relative), (3.5, True))

    def test_short_cue_time(self):
        self.assertEqual(parse_cue_time("01:02.000").lo, 62.0)

    def test_cue_time_is_not_a_clock_time(self):
        self.assertIsNone(p("00:00:03.500"))
        self.assertIsNone(parse_cue_time("21:16"))


class Garbage(unittest.TestCase):
    def test_non_times_return_none(self):
        for bad in [None, "", "soon", "~21:16", "about 21:14", True, "21:16 or so", "2026-03-02"]:
            with self.subTest(bad=bad):
                self.assertIsNone(p(bad))

    def test_offsets(self):
        self.assertEqual(parse_offset("-05:00"), EST)
        self.assertEqual(parse_offset("-0500"), EST)
        self.assertIsNone(parse_offset("EST"))


if __name__ == "__main__":
    unittest.main()
