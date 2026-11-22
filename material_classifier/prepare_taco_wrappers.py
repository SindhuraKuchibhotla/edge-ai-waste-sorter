"""Add TACO crisp packets and plastic wrappers as trash training examples.

This keeps source images and cropped examples in the ignored data directory.
The reserved TACO test crops are never copied into the model's train, validation,
or mixed test split. A split is assigned by source image ID before cropping.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


TACO_WRAPPER_LABELS = {"Crisp packet", "Other plastic wrapper"}
ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"


def download(url: str, target: Path) -> bool:
    if target.is_file():
        return True
    try:
        request = Request(url, headers={"User-Agent": "material-classifier/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        with Image.open(target) as image:
            image.verify()
        return True
    except Exception as exc:
        target.unlink(missing_ok=True)
        print(f"Could not download {url}: {exc}", flush=True)
        return False


def crop_annotation(image: Image.Image, source: dict, annotation: dict) -> Image.Image:
    x, y, width, height = annotation["bbox"]
    scale_x = image.width / source["width"]
    scale_y = image.height / source["height"]
    pad_x, pad_y = width * 0.12, height * 0.12
    box = (max(0, int((x - pad_x) * scale_x)),
           max(0, int((y - pad_y) * scale_y)),
           min(image.width, int((x + width + pad_x) * scale_x)),
           min(image.height, int((y + height + pad_y) * scale_y)))
    return image.crop(box).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="existing seven-class mixed dataset with explicit splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="local TACO annotations.json; downloaded when omitted")
    parser.add_argument("--min-short-side", type=int, default=40,
                        help="smallest projected object side in the 640px source image")
    args = parser.parse_args()
    if args.min_short_side < 1:
        parser.error("--min-short-side must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = args.annotations or args.output_dir / "annotations.json"
    if not annotations_file.is_file():
        if not download(ANNOTATIONS_URL, annotations_file):
            parser.error(f"Could not download TACO annotations to {annotations_file}")
    data = json.loads(annotations_file.read_text())
    categories = {category["id"]: category["name"] for category in data["categories"]}
    target_ids = {category["id"] for category in data["categories"]
                  if category["name"] in TACO_WRAPPER_LABELS}
    sources = {image["id"]: image for image in data["images"]}
    selected = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] not in target_ids:
            continue
        source = sources[annotation["image_id"]]
        if not source.get("flickr_640_url"):
            continue
        box = annotation["bbox"]
        projected_short = min(box[2], box[3]) * 640 / source["width"]
        if projected_short < args.min_short_side:
            continue
        previous = selected.get(source["id"])
        if previous is None or box[2] * box[3] > previous["bbox"][2] * previous["bbox"][3]:
            selected[source["id"]] = annotation

    image_ids = sorted(selected)
    random.Random(42).shuffle(image_ids)
    test_ids = set(image_ids[:max(1, round(len(image_ids) * 0.25))])
    for split in ("train", "validation", "test"):
        for class_dir in (args.source_dir / split).iterdir():
            if not class_dir.is_dir():
                continue
            target = args.output_dir / "mixed" / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for path in class_dir.iterdir():
                if path.is_file():
                    link = target / path.name
                    if not link.exists():
                        link.symlink_to(path.resolve())

    cache_dir = args.output_dir / "source_images"
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download, sources[image_id]["flickr_640_url"],
                                   cache_dir / f"{image_id}.jpg"): image_id
                   for image_id in image_ids}
        downloaded = {futures[future] for future in as_completed(futures) if future.result()}

    manifest = []
    for image_id in image_ids:
        if image_id not in downloaded:
            continue
        split = "test" if image_id in test_ids else "train"
        source = sources[image_id]
        target = (args.output_dir / "taco_test" if split == "test"
                  else args.output_dir / "mixed" / "train" / "trash") / f"taco_wrapper_{image_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(cache_dir / f"{image_id}.jpg") as image:
            crop_annotation(image, source, selected[image_id]).save(target, quality=95)
        manifest.append({"image_id": image_id, "split": split,
                         "category": categories[selected[image_id]["category_id"]],
                         "url": source["flickr_640_url"], "crop": str(target)})

    (args.output_dir / "taco_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"TACO wrappers: {sum(row['split'] == 'train' for row in manifest)} train, "
          f"{sum(row['split'] == 'test' for row in manifest)} reserved test; "
          f"{len(image_ids) - len(downloaded)} download failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
