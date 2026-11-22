"""Review saved camera frames and copy them into labeled belt-data folders."""

import argparse
from pathlib import Path
import shutil
import sys

import cv2

from .preview import DEFAULT_CAPTURES


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "belt-labeled"
DEFAULT_CLASSES = ("cardboard", "food_organic", "glass", "metal", "paper", "plastic", "trash")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_labeled(source: Path, output: Path, split: str, label: str) -> Path:
    """Copy one capture once; prevent accidentally assigning it to two splits."""
    existing = list(output.glob(f"*/*/{source.name}"))
    if existing:
        raise FileExistsError(f"{source.name} is already labeled at {existing[0]}")
    destination = output / split / label / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def fit_preview(frame, max_width=1280, max_height=800):
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True,
                        help="choose one split for this physical-item capture session")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="labels to show; use the same list for every split")
    args = parser.parse_args()
    if len(args.classes) > 9 or len(set(args.classes)) != len(args.classes):
        parser.error("provide 1-9 distinct class names")
    if not args.captures.is_dir():
        parser.error(f"capture folder does not exist: {args.captures}")
    files = sorted(path for path in args.captures.iterdir()
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"No images in {args.captures}")
        return 0
    keys = {ord(str(index)): label for index, label in enumerate(args.classes, 1)}
    print("Labels:", ", ".join(f"{index}={label}" for index, label in enumerate(args.classes, 1)))
    print("n=skip, q=quit. Capture files are copied and kept in place.")
    try:
        for path in files:
            if list(args.output.glob(f"*/*/{path.name}")):
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Cannot read {path}; skipping", file=sys.stderr)
                continue
            cv2.imshow("Label belt captures", fit_preview(frame))
            print(f"Label {path.name} for {args.split}:", flush=True)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("n"):
                    break
                if key in keys:
                    destination = copy_labeled(path, args.output, args.split, keys[key])
                    print(f"  -> {destination}", flush=True)
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
