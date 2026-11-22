"""Run the TrashNet material classifier on one centered item or a still image."""

import argparse
from datetime import datetime
from pathlib import Path
import platform
import sys

import cv2
import torch
from PIL import Image

from .model import DEFAULT_CHECKPOINT, best_device, image_transform, load_checkpoint


DEFAULT_CAPTURES = Path(__file__).resolve().parent / "captures"


def center_crop(frame, fraction: float, center_x: float = 0.5, center_y: float = 0.5):
    height, width = frame.shape[:2]
    side = max(1, round(min(height, width) * fraction))
    left = max(0, min(width - side, round(width * center_x - side / 2)))
    top = max(0, min(height - side, round(height * center_y - side / 2)))
    return frame[top:top + side, left:left + side], (left, top, left + side, top + side)


def predict(frame, model, device, transform, classes, roi_fraction,
            center_x: float = 0.5, center_y: float = 0.5):
    crop, box = center_crop(frame, roi_fraction, center_x, center_y)
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(tensor).softmax(dim=1)[0].cpu().tolist()
    ranked = sorted(zip(classes, scores), key=lambda pair: pair[1], reverse=True)
    return ranked, box


def annotate(frame, ranked, box):
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 210, 255), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 106), (0, 0, 0), -1)
    cv2.putText(frame, "One item inside box | s: save frame | q: quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    for index, (label, confidence) in enumerate(ranked[:3]):
        cv2.putText(frame, f"{label}: {confidence:.1%}", (10, 50 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 170) if index == 0 else (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def save_frame(frame, folder: Path, prefix: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot save camera frame: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    source.add_argument("--image", type=Path, help="classify a still image instead")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--roi", type=float, default=0.9,
                        help="fraction of short camera dimension inside square crop")
    parser.add_argument("--roi-center-x", type=float, default=0.5,
                        help="horizontal crop center as a fraction of frame width")
    parser.add_argument("--roi-center-y", type=float, default=0.5,
                        help="vertical crop center as a fraction of frame height")
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURES,
                        help="save full frames here with the s key for later labeling")
    parser.add_argument("--once", action="store_true",
                        help="capture, classify, and save one camera frame without a preview window")
    args = parser.parse_args()
    if not 0 < args.roi <= 1 or not 0 <= args.roi_center_x <= 1 or not 0 <= args.roi_center_y <= 1:
        parser.error("roi must be in (0, 1]; roi centers must be in [0, 1]")
    if args.once and args.image is not None:
        parser.error("--once requires a camera, not --image")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}; train the model first")

    device = best_device()
    model, classes = load_checkpoint(args.checkpoint, device)
    transform = image_transform(training=False)
    print(f"Loaded {args.checkpoint} on {device}; labels={classes}")
    print("This classifier always chooses a class, even when the view is empty.")

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"Cannot read image: {args.image}", file=sys.stderr)
            return 1
        ranked, _ = predict(frame, model, device, transform, classes, args.roi,
                            args.roi_center_x, args.roi_center_y)
        for label, confidence in ranked:
            print(f"{label}: {confidence:.3f}")
        return 0

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        print(f"Cannot open camera {args.camera}. Check camera permissions and try another index.",
              file=sys.stderr)
        return 1
    if args.once:
        print("Camera open. Capturing one frame.")
    else:
        print("Camera open. Press s to save an unlabeled belt frame, q or Esc to quit.")
    try:
        if args.once:
            frame = None
            for _ in range(5):
                okay, frame = camera.read()
                if not okay:
                    print("Camera did not return a frame.", file=sys.stderr)
                    return 1
            ranked, _ = predict(frame, model, device, transform, classes, args.roi,
                                args.roi_center_x, args.roi_center_y)
            path = save_frame(frame, args.capture_dir, "snapshot")
            print(f"Saved camera frame: {path}")
            for label, confidence in ranked:
                print(f"{label}: {confidence:.3f}")
            return 0
        while True:
            okay, frame = camera.read()
            if not okay:
                print("Camera stopped returning frames.", file=sys.stderr)
                return 1
            ranked, box = predict(frame, model, device, transform, classes, args.roi,
                                  args.roi_center_x, args.roi_center_y)
            cv2.imshow("TrashNet material preview", annotate(frame.copy(), ranked, box))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_frame(frame, args.capture_dir, "belt")
                print(f"Saved {path}", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
