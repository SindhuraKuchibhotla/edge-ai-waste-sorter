import json
from math import inf, nan
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from material_classifier.model import FOOD_ORGANIC_CHECKPOINT, image_transform, load_checkpoint
from material_classifier.package_uno_q import APP_FILES, package
from uno_q_material_logger.python.display import DisplayPublisher
from uno_q_material_logger.python.material_logger import (
    Config,
    MaterialClassifier,
    MaterialLogger,
    ResultWriter,
    crop_region,
    lane_for,
    prepare_input,
)


APP_ROOT = Path(__file__).resolve().parent.parent / "uno_q_material_logger"
CONFIG = Config.from_file(APP_ROOT / "config.json")
MODEL = APP_ROOT / "model" / "trashnet-mobilenetv3-small.onnx"
LABELS = APP_ROOT / "model" / "labels.json"


class LaneTests(unittest.TestCase):
    def test_all_seven_materials_map_to_three_lanes_at_any_score(self):
        expected = {
            "cardboard": "recycle", "glass": "recycle", "metal": "recycle",
            "paper": "recycle", "plastic": "trash", "food_organic": "compost",
            "trash": "trash",
        }
        self.assertEqual(set(json.loads(LABELS.read_text())["labels"]), set(expected))
        for material, lane in expected.items():
            with self.subTest(material=material):
                for confidence in (0.01, 0.40, 0.95):
                    self.assertEqual(lane_for(material, confidence), lane)
        self.assertEqual(lane_for("unrecognized", 0.9), "trash")
        for confidence in (nan, inf, -inf):
            with self.assertRaises(ValueError):
                lane_for("plastic", confidence)


class ExportParityTests(unittest.TestCase):
    def test_onnx_matches_trained_checkpoint_and_preprocessing(self):
        frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        crop = crop_region(frame, CONFIG)
        reference_input = image_transform(training=False)(
            Image.fromarray(crop[:, :, ::-1])
        ).unsqueeze(0).numpy()
        board_input = prepare_input(frame, CONFIG)
        self.assertTrue(np.array_equal(board_input, reference_input))

        model, labels = load_checkpoint(FOOD_ORGANIC_CHECKPOINT, torch.device("cpu"))
        with torch.inference_mode():
            reference = model(torch.from_numpy(reference_input)).softmax(dim=1).numpy()[0]
        board = MaterialClassifier(MODEL, LABELS, CONFIG)
        logits = board.session.run(None, {board.input_name: board_input})[0][0]
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        self.assertEqual(board.labels, labels)
        self.assertEqual(int(scores.argmax()), int(reference.argmax()))
        self.assertLess(float(np.max(np.abs(scores - reference))), 1e-4)


class CameraRecoveryTests(unittest.TestCase):
    def test_nonfinite_model_score_records_error_without_a_lane(self):
        class FakeCamera:
            def start(self):
                pass

            def capture(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        class FakeClassifier:
            def predict(self, _frame):
                return "plastic", nan

        with tempfile.TemporaryDirectory() as directory:
            logger = MaterialLogger(config=CONFIG, camera_factory=FakeCamera,
                                    classifier_factory=FakeClassifier,
                                    writer=ResultWriter(Path(directory)))
            result = logger.sample()
        self.assertEqual(result["camera_status"], "ok")
        self.assertIsNone(result["material"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["lane"])
        self.assertTrue(result["error"].startswith("model_error: ValueError:"))

    def test_unavailable_camera_recovers_and_logs_failure_without_fake_prediction(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        attempts = 0

        class FakeCamera:
            def __init__(self, fail_read=False):
                self.fail_read = fail_read
                self.reads = 0

            def start(self):
                pass

            def capture(self):
                self.reads += 1
                if self.fail_read and self.reads > 1:
                    raise OSError("camera unplugged")
                return frame

            def stop(self):
                pass

        class FakeClassifier:
            def predict(self, _frame):
                return "paper", 0.8

        def camera_factory():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("no USB camera")
            return FakeCamera(fail_read=attempts == 2)

        with tempfile.TemporaryDirectory() as directory:
            writer = ResultWriter(Path(directory))
            logger = MaterialLogger(config=CONFIG, camera_factory=camera_factory,
                                    classifier_factory=FakeClassifier, writer=writer)
            logger.started_at = 0
            at = datetime(2026, 9, 22, tzinfo=timezone.utc)
            results = [logger.sample(now=t, timestamp=at) for t in (0, 1, 5, 6, 11)]
            for result in results:
                writer.append(result)
            self.assertEqual([r["camera_status"] for r in results],
                             ["unavailable", "unavailable", "ok", "read_error", "ok"])
            self.assertEqual([r["material"] for r in results],
                             [None, None, "paper", None, "paper"])
            self.assertEqual([r["lane"] for r in results],
                             [None, None, "recycle", None, "recycle"])
            self.assertIsInstance(results[2]["inference_ms"], int)
            self.assertEqual(attempts, 3)
            lines = writer.path.read_text().splitlines()
            self.assertEqual([json.loads(line) for line in lines], results)

    def test_each_start_gets_its_own_file_even_at_the_same_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            started_at = datetime(2026, 9, 22, tzinfo=timezone.utc)
            first = ResultWriter(Path(directory), started_at=started_at)
            first.append({"timestamp_utc": "2026-09-22T23:59:59Z"})
            first.append({"timestamp_utc": "2026-09-23T00:00:01Z"})
            second = ResultWriter(Path(directory), started_at=started_at)
            second.append({"timestamp_utc": "2026-09-23T00:00:02Z"})
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(len(list(Path(directory).glob("results-*.txt"))), 2)
            self.assertEqual(len(first.path.read_text().splitlines()), 2)
            self.assertEqual(len(second.path.read_text().splitlines()), 1)


class DisplayPublisherTests(unittest.TestCase):
    def test_bridge_arguments_and_status_changes(self):
        sent = []
        publisher = DisplayPublisher(lambda method, *args: sent.append((method, args)))
        publisher.publish({"camera_status": "unavailable", "error": "no camera"})
        publisher.publish({"camera_status": "unavailable", "error": "no camera"})
        publisher.publish({"camera_status": "ok", "error": None,
                           "material": "cardboard", "confidence": 0.736, "lane": "recycle",
                           "inference_ms": 38})
        publisher.publish({"camera_status": "ok", "error": None,
                           "material": "food_organic", "confidence": 0.812, "lane": "compost",
                           "inference_ms": 42})
        publisher.publish({"camera_status": "ok", "error": None,
                           "material": "plastic", "confidence": 0.399, "lane": "trash",
                           "inference_ms": 39})
        publisher.publish({"camera_status": "ok", "error": "model_error: failed"})
        self.assertEqual(sent, [
            ("show_status", ("NO CAMERA",)),
            ("show_result", ("RECYCLE", "cardboard", 74, 38)),
            ("show_result", ("COMPOST", "food_organic", 81, 42)),
            ("show_result", ("TRASH", "plastic", 40, 39)),
            ("show_status", ("MODEL ERROR",)),
        ])

    def test_bridge_failure_does_not_stop_file_logging(self):
        class FakeCamera:
            def start(self):
                pass

            def capture(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        class FakeClassifier:
            def predict(self, _frame):
                return "paper", 0.8

        def failed_notify(_method, *_args):
            raise ConnectionError("router unavailable")

        with tempfile.TemporaryDirectory() as directory:
            writer = ResultWriter(Path(directory))
            logger = MaterialLogger(config=CONFIG, camera_factory=FakeCamera,
                                    classifier_factory=FakeClassifier, writer=writer,
                                    publisher=DisplayPublisher(failed_notify))
            with patch("uno_q_material_logger.python.material_logger.sleep", return_value=None):
                logger.loop()
            saved = json.loads(writer.path.read_text().splitlines()[0])
            self.assertEqual(saved["material"], "paper")
            self.assertEqual(saved["confidence"], 0.8)
            self.assertEqual(saved["lane"], "recycle")


class PackageTests(unittest.TestCase):
    def test_zip_has_importable_root_and_no_local_captures(self):
        from zipfile import ZipFile

        with tempfile.TemporaryDirectory() as directory:
            archive = package(Path(directory) / "app.zip")
            with ZipFile(archive) as zipped:
                self.assertEqual(set(zipped.namelist()), set(APP_FILES))
                self.assertIn("app.yaml", zipped.namelist())
                self.assertIn("sketch/libraries/Adafruit_GFX_Library/Adafruit_GFX.h", zipped.namelist())
                self.assertIn("sketch/libraries/Adafruit_BusIO/Adafruit_I2CDevice.h", zipped.namelist())
                self.assertNotIn("data/", zipped.namelist())


if __name__ == "__main__":
    unittest.main()
