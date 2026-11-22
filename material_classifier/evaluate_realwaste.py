"""Evaluate a material checkpoint on UCI RealWaste without extracting its ZIP."""

import argparse
import csv
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image
import torch

from .model import best_device, image_transform, load_checkpoint


SOURCE_LABELS = {
    "Cardboard": "cardboard",
    "Food Organics": "food_organic",
    "Glass": "glass",
    "Metal": "metal",
    "Miscellaneous Trash": "trash",
    "Paper": "paper",
    "Plastic": "plastic",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--source-manifest", type=Path,
                        help="mixed dataset manifest; evaluate only its RealWaste test images")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    allowed_paths = None
    if args.source_manifest:
        split = json.loads(args.source_manifest.read_text())["splits"]["test"]
        allowed_paths = {entry["source_path"] for entry in split
                         if entry["source"] == "RealWaste"}
        if not allowed_paths:
            parser.error("source manifest contains no RealWaste test images")

    device = best_device()
    model, labels = load_checkpoint(args.checkpoint, device)
    transform = image_transform(training=False)
    rows = []
    batch = []

    def infer_batch() -> None:
        if not batch:
            return
        with torch.inference_mode():
            probabilities = model(torch.stack([item[1] for item in batch]).to(device))
            probabilities = probabilities.softmax(dim=1).cpu().tolist()
        for (row, _), scores in zip(batch, probabilities):
            index = max(range(len(scores)), key=scores.__getitem__)
            row["predicted"] = labels[index]
            row["confidence"] = round(scores[index], 6)
            rows.append(row)
        batch.clear()

    with zipfile.ZipFile(args.archive) as archive:
        paths = sorted(name for name in archive.namelist()
                       if name.startswith("realwaste-main/RealWaste/")
                       and name.lower().endswith((".jpg", ".jpeg", ".png")))
        for position, path in enumerate(paths, 1):
            if allowed_paths is not None and path not in allowed_paths:
                continue
            category = path.split("/")[2]
            expected = SOURCE_LABELS.get(category)
            if expected not in labels:
                continue
            with archive.open(path) as handle:
                image = Image.open(BytesIO(handle.read())).convert("RGB")
            batch.append(({"path": path, "source_category": category,
                           "expected": expected}, transform(image)))
            if len(batch) == args.batch_size:
                infer_batch()
            if position % 500 == 0:
                print(f"Read {position}/{len(paths)} source images", flush=True)
        infer_batch()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "source_category", "expected",
                                                       "predicted", "confidence"))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["expected"] for row in rows)
    predicted_counts = Counter(row["predicted"] for row in rows)
    correct = Counter(row["expected"] for row in rows
                      if row["predicted"] == row["expected"])
    per_class = {label: {"images": counts[label], "correct": correct[label],
                         "recall": round(correct[label] / counts[label], 4),
                         "precision": round(correct[label] / predicted_counts[label], 4)
                         if predicted_counts[label] else None}
                 for label in sorted(counts)}
    total_correct = sum(correct.values())
    report = {"dataset": "UCI RealWaste (id 908)", "checkpoint": str(args.checkpoint),
              "source_manifest": str(args.source_manifest) if args.source_manifest else None,
              "labels": labels, "images": len(rows), "correct": total_correct,
              "accuracy": round(total_correct / len(rows), 4),
              "macro_recall": round(sum(item["recall"] for item in per_class.values()) /
                                    len(per_class), 4),
              "per_class": per_class,
              "confusion": {label: dict(Counter(row["predicted"] for row in rows
                                                if row["expected"] == label))
                            for label in sorted(counts)}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("images", "correct", "accuracy",
                                                     "macro_recall", "per_class")}, indent=2))


if __name__ == "__main__":
    main()
