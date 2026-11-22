import tempfile
import unittest
from pathlib import Path

import numpy as np

from uno_q_material_logger.python.display import DisplayPublisher
from uno_q_material_logger.python.material_logger import (
    BeltGate,
    Config,
    MaterialLogger,
    ResultWriter,
)


APP_ROOT = Path(__file__).resolve().parent.parent / "uno_q_material_logger"


class BeltGateTests(unittest.TestCase):
    def test_learns_empty_belt_then_sees_an_item(self):
        gate = BeltGate(threshold=12, learn_frames=3)
        belt = np.full((100, 100, 3), 120, np.uint8)
        item = belt.copy()
        item[20:80, 20:80] = 250
        self.assertFalse(any(gate.update(belt) for _ in range(5)))
        self.assertTrue(gate.update(item))
        self.assertFalse(gate.update(belt))


class EmptyBeltPublishingTests(unittest.TestCase):
    def test_only_items_reach_the_oled_and_each_item_starts_with_scanning(self):
        belt = np.full((480, 640, 3), 120, np.uint8)
        item = belt.copy()
        item[100:380, 200:440] = 240
        frames = [belt] * 4 + [item] * 2 + [belt] * 2 + [item] * 2 + [belt]

        class Camera:
            def start(self): pass
            def stop(self): pass
            def capture(self): return frames.pop(0)

        class Classifier:
            def predict(self, frame):
                return ("metal", 0.9) if frame.mean() > 130 else ("cardboard", 0.7)

        sent = []
        config = Config.from_file(APP_ROOT / "config.json")
        self.assertTrue(config.detect_empty_belt)
        with tempfile.TemporaryDirectory() as directory:
            logger = MaterialLogger(config=config, camera_factory=Camera, classifier_factory=Classifier,
                                    writer=ResultWriter(Path(directory)),
                                    publisher=DisplayPublisher(lambda m, *a: sent.append((m, a[:2]))))
            for t in range(11):
                logger.publisher.publish(logger.sample(now=float(t)))
        self.assertEqual(sent, [
            ("display_scanning", ()), ("show_result", ("RECYCLE", "metal")), ("show_result", ("RECYCLE", "metal")),
            ("display_scanning", ()), ("show_result", ("RECYCLE", "metal")), ("show_result", ("RECYCLE", "metal")),
        ])

    def test_check_can_be_switched_off(self):
        values = {"camera_resolution": [640, 480], "roi_fraction": 0.9, "roi_center_x": 0.5,
                  "roi_center_y": 0.5, "detect_empty_belt": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(__import__("json").dumps(values))
            self.assertFalse(Config.from_file(path).detect_empty_belt)


if __name__ == "__main__":
    unittest.main()
