"""Local camera preview and still-image check for the UNO Q prototype."""

import argparse
from pathlib import Path
import sys
from time import monotonic

import cv2

from python.sorting import DetectionVoter, lane_for
from .yolox import Detection, YoloXNano


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "yolox_nano.onnx"
COLORS = {"compost": (40, 180, 40), "recycle": (230, 120, 30), "trash": (40, 40, 230)}


def draw(frame, detections: list[Detection], lane: str = "clear"):
    for result in detections:
        color = COLORS[lane_for(result.label)]
        x1, y1, x2, y2 = result.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{result.label} {result.confidence:.2f} -> {lane_for(result.label)}"
        text_y = max(20, y1 - 8)
        cv2.putText(frame, caption, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"stable lane: {lane}   q: quit", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def voter_input(detections: list[Detection]) -> dict:
    grouped = {}
    for result in detections:
        grouped.setdefault(result.label, []).append({"confidence": result.confidence})
    return grouped


def run_camera(index: int, model: YoloXNano, voter: DetectionVoter) -> None:
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera {index}. Check its connection, close other camera apps, "
            "and allow camera access for your Terminal or Python app in macOS Settings."
        )
    print(f"Camera {index} opened. Press q or Esc in the preview to quit.", flush=True)
    lane = "clear"
    failed_reads = 0
    try:
        while True:
            okay, frame = camera.read()
            if not okay:
                failed_reads += 1
                if failed_reads >= 10:
                    raise RuntimeError("Camera stopped returning frames; check the USB connection.")
                continue
            failed_reads = 0
            detections = model.detect(frame)
            now = monotonic()
            decision = voter.ingest(voter_input(detections), now)
            if decision is None:
                decision = voter.expire(now)
            if decision is not None:
                lane = decision.lane
                print(decision, flush=True)
            cv2.imshow("Waste sorter - YOLOX Nano", draw(frame, detections, lane))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_image(path: Path, output: Path, model: YoloXNano) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    detections = model.detect(frame)
    for result in detections:
        print(f"object={result.label} lane={lane_for(result.label)} "
              f"model_confidence={result.confidence:.2f} box={result.box}")
    if not detections:
        print("No objects detected above the confidence threshold.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw(frame, detections)):
        raise RuntimeError(f"Cannot write annotated image: {output}")
    print(f"Annotated image: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, metavar="INDEX", help="macOS camera index (default: 0)")
    source.add_argument("--image", type=Path, help="run once on a still image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOX Nano COCO ONNX file")
    parser.add_argument("--output", type=Path,
                        default=Path("local_test/output/annotated.jpg"),
                        help="annotated image path for --image")
    parser.add_argument("--confidence", type=float, default=0.5, help="minimum detection score")
    parser.add_argument("--window-s", type=float, default=1.0, help="vote window in seconds")
    parser.add_argument("--min-votes", type=int, default=3, help="votes required to change label")
    parser.add_argument("--clear-s", type=float, default=1.5, help="seconds without detection to clear")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.window_s <= 0 or args.min_votes < 1 or args.clear_s <= 0:
        parser.error("confidence must be in (0, 1], window-s and clear-s positive, and min-votes >= 1")
    if not args.model.is_file():
        parser.error(f"model missing: {args.model}. Download it using the README command.")
    try:
        model = YoloXNano(str(args.model), args.confidence)
        if args.image:
            run_image(args.image, args.output, model)
        else:
            voter = DetectionVoter(args.window_s, args.min_votes, args.clear_s)
            run_camera(args.camera if args.camera is not None else 0, model, voter)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
