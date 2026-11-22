"""Check independent TACO wrapper crops and optional belt photos."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image

from .model import best_device, image_transform, load_checkpoint
from .preview import predict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="reserved TACO wrapper crops from prepare_taco_wrappers")
    parser.add_argument("--belt-images", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.test_dir.glob("*.jpg"))
    if not paths:
        parser.error(f"No test JPGs in {args.test_dir}")
    device = best_device()
    model, classes = load_checkpoint(args.checkpoint, device)
    transform = image_transform(training=False)
    count = Counter()
    rows = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            scores = model(tensor).softmax(dim=1)[0].cpu().tolist()
            ranked = sorted(zip(classes, scores), key=lambda item: item[1], reverse=True)
            count[ranked[0][0]] += 1
            rows.append({"image": path.name, "predictions": dict(ranked)})
    belts = []
    for path in args.belt_images:
        frame = cv2.imread(str(path))
        if frame is None:
            parser.error(f"Cannot read belt image: {path}")
        ranked, box = predict(frame, model, device, transform, classes, 0.9)
        belts.append({"image": path.name, "roi_box": box, "predictions": dict(ranked)})
    result = {"checkpoint": str(args.checkpoint), "taco_test_images": len(paths),
              "taco_predictions": dict(sorted(count.items())),
              "taco_trash_recall": count["trash"] / len(paths),
              "taco_rows": rows, "belt": belts}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"TACO wrapper recall: {count['trash']}/{len(paths)}; predictions={dict(count)}")
    for row in belts:
        top = max(row["predictions"].items(), key=lambda item: item[1])
        print(f"{row['image']}: {top[0]} {top[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
