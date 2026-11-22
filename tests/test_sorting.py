import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from sorting import DetectionVoter, best_detection, lane_for


def detections(label, confidence=0.9):
    return {label: [{"confidence": confidence}]}


class DetectionVoterTests(unittest.TestCase):
    def test_lane_rules_and_unmapped_trash_fallback(self):
        self.assertEqual(lane_for(" Cup "), "compost")
        self.assertEqual(lane_for("bottle"), "recycle")
        self.assertEqual(lane_for("banana"), "trash")
        self.assertEqual(lane_for("apple"), "trash")

    def test_uses_highest_confidence_detection(self):
        self.assertEqual(
            best_detection({"banana": [{"confidence": 0.6}], "bottle": [{"confidence": 0.8}]}),
            ("bottle", 0.8),
        )
        self.assertIsNone(
            best_detection({"banana": [{"confidence": float("nan")}]})
        )

    def test_votes_outside_window_do_not_accumulate(self):
        voter = DetectionVoter()
        for time in (0.0, 1.1, 2.2):
            self.assertIsNone(voter.ingest(detections("banana"), time))
        self.assertIsNone(voter.expire(3.0))

    def test_vote_resists_one_bad_frame_and_reports_mean_score(self):
        voter = DetectionVoter()
        self.assertIsNone(voter.ingest(detections("cup", 0.8), 0.0))
        self.assertIsNone(voter.ingest(detections("cup", 0.9), 0.1))
        first = voter.ingest(detections("cup", 1.0), 0.2)
        self.assertEqual(str(first), "object=cup lane=compost model_confidence=0.90")
        self.assertIsNone(voter.ingest(detections("bottle"), 0.3))
        self.assertIsNone(voter.ingest(detections("cup"), 0.4))

    def test_new_stable_label_prints_then_clears_once(self):
        voter = DetectionVoter()
        for time in (0.0, 0.1):
            self.assertIsNone(voter.ingest(detections("bottle"), time))
        self.assertEqual(voter.ingest(detections("bottle"), 0.2).lane, "recycle")
        self.assertIsNone(voter.ingest(detections("cup"), 1.3))
        self.assertIsNone(voter.ingest(detections("cup"), 1.4))
        self.assertEqual(voter.ingest(detections("cup"), 1.5).lane, "compost")
        self.assertIsNone(voter.expire(2.9))
        self.assertEqual(str(voter.expire(3.0)), "object=none lane=clear model_confidence=none")
        self.assertIsNone(voter.expire(4.0))


if __name__ == "__main__":
    unittest.main()
