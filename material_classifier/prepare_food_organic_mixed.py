"""Add deduplicated UCI RealWaste images to the seven-class TrashNet/King splits."""

import argparse
import json
from pathlib import Path
import random
import zipfile

from .prepare_food_organic import MODULE_DIR, SPLIT_NAMES, deduplicate, split_paths


DEFAULT_BASE = MODULE_DIR / "data" / "food-organic-seven-class-v3"
DEFAULT_REALWASTE = MODULE_DIR / "data" / "realwaste.zip"
DEFAULT_OUTPUT = MODULE_DIR / "data" / "food-organic-mixed"
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
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--realwaste-archive", type=Path, default=DEFAULT_REALWASTE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a fresh directory")

    rng = random.Random(args.seed)
    manifest = {name: [] for name in SPLIT_NAMES}
    base_classes = set()
    for split in SPLIT_NAMES:
        for class_dir in sorted((args.base_data / split).iterdir()):
            if not class_dir.is_dir():
                continue
            base_classes.add(class_dir.name)
            for source in sorted(class_dir.iterdir()):
                if not source.is_file():
                    continue
                target = args.output / split / class_dir.name / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
                manifest[split].append({"path": str(target.relative_to(args.output)),
                                        "source": "TrashNet/King",
                                        "source_path": str(source.resolve())})
    if base_classes != set(SOURCE_LABELS.values()):
        parser.error(f"base classes do not match RealWaste mapping: {base_classes}")

    counts = {}
    with zipfile.ZipFile(args.realwaste_archive) as archive:
        for category, label in SOURCE_LABELS.items():
            prefix = f"realwaste-main/RealWaste/{category}/"
            original_paths = [name for name in archive.namelist()
                              if name.startswith(prefix) and name.lower().endswith(".jpg")]
            if not original_paths:
                parser.error(f"RealWaste category missing: {category}")
            unique_paths, unique_count = deduplicate(archive, original_paths, rng)
            counts[category] = {"source_images": len(original_paths),
                                "unique_images": unique_count}
            for split, selected in split_paths(unique_paths, rng).items():
                for source_path in selected:
                    target = (args.output / split / label /
                              f"realwaste_{Path(source_path).name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(source_path))
                    manifest[split].append({"path": str(target.relative_to(args.output)),
                                            "source": "RealWaste",
                                            "source_path": source_path})

    (args.output / "source_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "base_data": str(args.base_data),
        "realwaste_archive": str(args.realwaste_archive),
        "realwaste_categories": counts,
        "splits": manifest,
    }, indent=2) + "\n")
    for split, entries in manifest.items():
        realwaste = sum(item["source"] == "RealWaste" for item in entries)
        print(f"{split}: {len(entries)} images, including {realwaste} RealWaste")
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
