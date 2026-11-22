import tempfile
import unittest
from pathlib import Path

from material_classifier.label_captures import copy_labeled


class CopyLabeledTests(unittest.TestCase):
    def test_preserves_capture_and_rejects_second_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "capture.jpg"
            capture.write_bytes(b"camera bytes")
            output = root / "belt-labeled"
            destination = copy_labeled(capture, output, "train", "glass")
            self.assertEqual(destination.read_bytes(), capture.read_bytes())
            self.assertTrue(capture.is_file())
            with self.assertRaises(FileExistsError):
                copy_labeled(capture, output, "test", "plastic")


if __name__ == "__main__":
    unittest.main()
