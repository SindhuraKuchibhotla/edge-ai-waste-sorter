import tempfile
import unittest
from pathlib import Path

from material_classifier.train import split_images


class BeltSplitTests(unittest.TestCase):
    def test_explicit_splits_keep_items_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "validation", "test"):
                for label in ("cardboard", "plastic"):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    (folder / f"{split}-{label}.jpg").touch()
            classes, splits = split_images(root, seed=42, max_per_class=None)
            self.assertEqual(classes, ["cardboard", "plastic"])
            self.assertEqual({key: len(value) for key, value in splits.items()},
                             {"train": 2, "validation": 2, "test": 2})
            self.assertEqual(len({path for items in splits.values() for path, _ in items}), 6)

    def test_partial_explicit_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "train").mkdir()
            with self.assertRaisesRegex(ValueError, "all of"):
                split_images(Path(temp), seed=42, max_per_class=None)


if __name__ == "__main__":
    unittest.main()
