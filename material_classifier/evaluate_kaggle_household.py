"""Evaluate the TrashNet material model on a balanced sample of King's dataset.

The source dataset has item labels, not our six material labels. Only categories
with an unambiguous material are scored; other categories are reported as
prediction distributions without assigning a synthetic ground truth.
"""

import argparse
import csv
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

from PIL import Image
import torch

from .model import REPO_CHECKPOINT, best_device, image_transform, load_checkpoint


MATERIAL_OF = {
    "aluminum_food_cans": "metal",
    "aluminum_soda_cans": "metal",
    "steel_food_cans": "metal",
    "aerosol_cans": "metal",
    "cardboard_boxes": "cardboard",
    "cardboard_packaging": "cardboard",
    "glass_beverage_bottles": "glass",
    "glass_cosmetic_containers": "glass",
    "glass_food_jars": "glass",
    "magazines": "paper",
    "newspaper": "paper",
    "office_paper": "paper",
    "disposable_plastic_cutlery": "plastic",
    "plastic_cup_lids": "plastic",
    "plastic_detergent_bottles": "plastic",
    "plastic_food_containers": "plastic",
    "plastic_shopping_bags": "plastic",
    "plastic_soda_bottles": "plastic",
    "plastic_straws": "plastic",
    "plastic_trash_bags": "plastic",
    "plastic_water_bottles": "plastic",
}


def summarize(rows):
    counts = Counter(row["predicted"] for row in rows)
    scored = [row for row in rows if row["expected"]]
    correct = sum(row["predicted"] == row["expected"] for row in scored)
    return {
        "images": len(rows),
        "scored": len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "predicted_counts": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_CHECKPOINT)
    parser.add_argument("--per-category-split", type=int, default=25,
                        help="sample this many images from each category and image style")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "kaggle-household")
    args = parser.parse_args()
    if args.per_category_split < 1 or args.batch_size < 1:
        parser.error("sample count and batch size must be positive")

    device = best_device()
    model, labels = load_checkpoint(args.checkpoint, device)
    transform = image_transform(training=False)
    rng = random.Random(args.seed)
    rows = []
    pending = []

    def infer_batch():
        if not pending:
            return
        tensors = torch.stack([tensor for _, tensor in pending]).to(device)
        with torch.inference_mode():
            scores = model(tensors).softmax(dim=1).cpu()
        for (row, _), probabilities in zip(pending, scores):
            index = int(probabilities.argmax())
            row["predicted"] = labels[index]
            row["confidence"] = round(float(probabilities[index]), 5)
            rows.append(row)
        pending.clear()

    with zipfile.ZipFile(args.archive) as archive:
        groups = defaultdict(list)
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 5 and parts[:2] == ["images", "images"] and name.lower().endswith(".png"):
                groups[(parts[2], parts[3])].append(name)
        for (category, split), paths in sorted(groups.items()):
            for path in rng.sample(sorted(paths), min(len(paths), args.per_category_split)):
                with archive.open(path) as handle:
                    image = Image.open(BytesIO(handle.read())).convert("RGB")
                row = {"path": path, "category": category, "split": split,
                       "expected": MATERIAL_OF.get(category, "")}
                pending.append((row, transform(image)))
                if len(pending) == args.batch_size:
                    infer_batch()
            print(f"{category}/{split}: {min(len(paths), args.per_category_split)} images", flush=True)
        infer_batch()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "category", "split", "expected", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    by_split = {split: summarize([row for row in rows if row["split"] == split])
                for split in sorted({row["split"] for row in rows})}
    by_material = {material: summarize([row for row in rows if row["expected"] == material])
                   for material in sorted(set(MATERIAL_OF.values()))}
    by_category = {category: summarize([row for row in rows if row["category"] == category])
                   for category in sorted({row["category"] for row in rows})}
    report = {
        "dataset": "alistairking/recyclable-and-household-waste-classification",
        "checkpoint": str(args.checkpoint), "labels": labels,
        "seed": args.seed, "per_category_split": args.per_category_split,
        "mapped_categories": MATERIAL_OF,
        "overall": summarize(rows), "by_split": by_split,
        "by_material": by_material, "by_category": by_category,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"overall": report["overall"], "by_split": by_split,
                      "by_material": by_material}, indent=2))
    print(f"Saved {args.output / 'summary.json'} and {args.output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
