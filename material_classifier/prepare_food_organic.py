"""Build seven-class splits from TrashNet and King's food-organic categories.

TrashNet retains its original split, so its held-out test photos remain unseen
by the starting checkpoint. Organic images are split within category and style.
The generated dataset lives under the ignored data directory.
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parent
TRASHNET_ROOT = MODULE_DIR / "data" / "dataset-resized"
TRASHNET_SPLIT = MODULE_DIR / "output" / "split.json"
KING_ARCHIVE = MODULE_DIR / "data" / "recyclable-household.zip"
DEFAULT_OUTPUT = MODULE_DIR / "data" / "food-organic-seven-class"
ORGANIC_CATEGORIES = ("food_waste", "eggshells", "coffee_grounds")
SPLIT_NAMES = ("train", "validation", "test")


def split_paths(paths: list[str], rng: random.Random) -> dict[str, list[str]]:
    ordered = sorted(paths)
    rng.shuffle(ordered)
    train_end = int(len(ordered) * 0.7)
    validation_end = int(len(ordered) * 0.85)
    return dict(zip(SPLIT_NAMES, (ordered[:train_end],
                                  ordered[train_end:validation_end],
                                  ordered[validation_end:])))


def deduplicate(archive: zipfile.ZipFile, paths: list[str], rng: random.Random):
    """Keep one image from each visually identical or nearly identical group."""
    items = []
    for path in sorted(paths):
        with archive.open(path) as handle:
            image = Image.open(BytesIO(handle.read())).convert("RGB")
        gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
        difference_hash = sum(int(gray[y, x] > gray[y, x + 1]) << (y * 8 + x)
                              for y in range(8) for x in range(8))
        thumbnail = np.asarray(image.resize((32, 32)), dtype=np.float32)
        items.append((path, difference_hash, thumbnail))

    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, (path, first_hash, first_thumb) in enumerate(items):
        category = path.split("/")[2]
        for j in range(i + 1, len(items)):
            other_path, other_hash, other_thumb = items[j]
            if other_path.split("/")[2] != category:
                continue
            if (first_hash ^ other_hash).bit_count() > 2:
                continue
            if float(np.mean((first_thumb - other_thumb) ** 2)) < 100:
                parent[find(j)] = find(i)

    groups = {}
    for index, (path, _, _) in enumerate(items):
        groups.setdefault(find(index), []).append(path)
    return [rng.choice(copies) for copies in groups.values()], len(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trashnet-root", type=Path, default=TRASHNET_ROOT)
    parser.add_argument("--trashnet-split", type=Path, default=TRASHNET_SPLIT)
    parser.add_argument("--king-archive", type=Path, default=KING_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a fresh directory")

    original = json.loads(args.trashnet_split.read_text())
    if set(original) != set(SPLIT_NAMES):
        parser.error(f"TrashNet split must contain {SPLIT_NAMES}")
    rng = random.Random(args.seed)
    manifest = {split: [] for split in SPLIT_NAMES}

    with zipfile.ZipFile(args.king_archive) as archive:
        organic_paths = []
        for name in archive.namelist():
            parts = name.split("/")
            if (len(parts) == 5 and parts[:2] == ["images", "images"]
                    and parts[2] in ORGANIC_CATEGORIES
                    and parts[3] in ("default", "real_world")
                    and name.lower().endswith(".png")):
                organic_paths.append(name)
        # The source archive repeats stock photos under different filenames,
        # encodings, and styles. Deduplicate before splitting to prevent
        # nearly identical photos from crossing train/validation/test.
        unique_paths, unique_count = deduplicate(archive, organic_paths, rng)
        organic_groups = {}
        for representative in unique_paths:
            parts = representative.split("/")
            organic_groups.setdefault((parts[2], parts[3]), []).append(representative)
        expected_groups = {(category, style) for category in ORGANIC_CATEGORIES
                           for style in ("default", "real_world")}
        if set(organic_groups) != expected_groups:
            parser.error(f"Missing organic source folders: {expected_groups - set(organic_groups)}")

        for split, entries in original.items():
            for entry in entries:
                source = args.trashnet_root / entry["path"]
                if not source.is_file():
                    parser.error(f"TrashNet image missing: {source}")
                target = args.output / split / entry["label"] / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
                manifest[split].append({"path": str(target.relative_to(args.output)),
                                        "source": "TrashNet", "source_path": entry["path"]})

        for (category, style), paths in sorted(organic_groups.items()):
            for split, selected in split_paths(paths, rng).items():
                for name in selected:
                    target = (args.output / split / "food_organic" /
                              f"king_{category}_{style}_{Path(name).name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                    manifest[split].append({"path": str(target.relative_to(args.output)),
                                            "source": "King", "source_path": name})

    (args.output / "source_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "trashnet_split": str(args.trashnet_split),
        "king_archive": str(args.king_archive),
        "organic_categories": ORGANIC_CATEGORIES,
        "excluded_organic_category": "tea_bags",
        "organic_source_images": len(organic_paths),
        "organic_unique_images": unique_count,
        "deduplication": "Within category, 64-bit dHash distance <= 2 and 32x32 RGB MSE < 100 before any split",
        "splits": manifest,
    }, indent=2) + "\n")
    for split, entries in manifest.items():
        food_count = sum(entry["source"] == "King" for entry in entries)
        print(f"{split}: {len(entries)} images, including {food_count} food_organic")
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
