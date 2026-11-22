"""Train a small ImageNet-pretrained classifier on Kaggle TrashNet or belt photos."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import best_device, build_model, image_transform


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "dataset-resized"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImagePaths(Dataset):
    def __init__(self, items: list[tuple[Path, int]], *, training: bool):
        self.items = items
        self.transform = image_transform(training=training)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def split_images(root: Path, seed: int, max_per_class: int | None):
    """Read explicit split folders, or split flat class folders 70/15/15."""
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    split_names = ("train", "validation", "test")
    existing_splits = [name for name in split_names if (root / name).is_dir()]
    if existing_splits:
        if len(existing_splits) != len(split_names):
            raise ValueError(f"Explicit dataset needs all of {split_names}; found {existing_splits}")
        if max_per_class is not None:
            raise ValueError("--max-per-class is only supported for flat class folders")
        classes = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())
        if len(classes) < 2:
            raise ValueError("Explicit train split needs at least two class folders")
        splits = {}
        for split_name in split_names:
            directory = root / split_name
            found = sorted(path.name for path in directory.iterdir() if path.is_dir())
            if found != classes:
                raise ValueError(f"{split_name} classes {found} do not match train classes {classes}")
            items = []
            for index, class_name in enumerate(classes):
                images = sorted(path for path in (directory / class_name).iterdir()
                                if path.is_file() and path.suffix.lower() in EXTENSIONS)
                if not images:
                    raise ValueError(f"No images in {directory / class_name}")
                items.extend((path, index) for path in images)
            splits[split_name] = items
        return classes, splits

    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected at least two class folders in {root}")
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for index, name in enumerate(classes):
        paths = sorted(path for path in (root / name).iterdir()
                       if path.is_file() and path.suffix.lower() in EXTENSIONS)
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        if len(paths) < 7:
            raise ValueError(f"Class {name!r} has only {len(paths)} images; need at least 7")
        train_end = int(len(paths) * 0.7)
        val_end = int(len(paths) * 0.85)
        for key, group in (("train", paths[:train_end]),
                           ("validation", paths[train_end:val_end]),
                           ("test", paths[val_end:])):
            splits[key].extend((path, index) for path in group)
    for items in splits.values():
        rng.shuffle(items)
    return classes, splits


def evaluate(model, loader, loss_function, device, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    confusion = [[0] * num_classes for _ in range(num_classes)]
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += loss_function(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            count += len(labels)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual][predicted] += 1
    recalls = [confusion[index][index] / sum(row) for index, row in enumerate(confusion)]
    return {"loss": total_loss / count, "accuracy": correct / count,
            "macro_recall": sum(recalls) / num_classes,
            "per_class_recall": recalls, "count": count,
            "confusion_matrix": confusion}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="flat class folders, or train/validation/test class folders")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int,
                        help="optional small smoke-test subset; default uses every image")
    parser.add_argument("--start-from", type=Path,
                        help="use a previous checkpoint as the starting point for belt photos")
    parser.add_argument("--unfreeze-last", type=int, default=0,
                        help="also train this many final backbone blocks (useful for belt photos)")
    parser.add_argument("--balance-classes", action="store_true",
                        help="sample each training class approximately equally")
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_recall"),
                        default="accuracy", help="validation metric used to save the best epoch")
    parser.add_argument("--train-samples-per-epoch", type=int,
                        help="balanced draws per epoch; defaults to the training image count")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.unfreeze_last < 0:
        parser.error("epochs and batch size must be positive; learning rate > 0; unfreeze-last >= 0")
    if args.max_per_class is not None and args.max_per_class < 7:
        parser.error("max-per-class must be at least 7")
    if args.train_samples_per_epoch is not None and args.train_samples_per_epoch < 1:
        parser.error("train-samples-per-epoch must be positive")
    if args.train_samples_per_epoch is not None and not args.balance_classes:
        parser.error("train-samples-per-epoch requires --balance-classes")

    torch.manual_seed(args.seed)
    classes, splits = split_images(args.data_dir, args.seed, args.max_per_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        name: [{"path": str(path.relative_to(args.data_dir)), "label": classes[label]}
               for path, label in items]
        for name, items in splits.items()
    }
    (args.output_dir / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    print("Classes:", classes, flush=True)
    for name, items in splits.items():
        print(f"{name}: {len(items)} {dict(Counter(classes[label] for _, label in items))}",
              flush=True)

    device = best_device()
    print("Device:", device, flush=True)
    model = build_model(len(classes), pretrained=args.start_from is None)
    if args.start_from is not None:
        source = torch.load(args.start_from, map_location="cpu", weights_only=True)
        previous_classes = source["classes"]
        if previous_classes == classes:
            model.load_state_dict(source["state_dict"])
        else:
            # Keep the trained features and old class rows when adding a class.
            feature_weights = {key: value for key, value in source["state_dict"].items()
                               if not key.startswith("classifier.3.")}
            model.load_state_dict(feature_weights, strict=False)
            previous_weight = source["state_dict"]["classifier.3.weight"]
            previous_bias = source["state_dict"]["classifier.3.bias"]
            with torch.no_grad():
                for index, name in enumerate(classes):
                    if name in previous_classes:
                        old_index = previous_classes.index(name)
                        model.classifier[-1].weight[index].copy_(previous_weight[old_index])
                        model.classifier[-1].bias[index].copy_(previous_bias[old_index])
            print(f"New labels {classes}; retained output rows for "
                  f"{sorted(set(previous_classes) & set(classes))}", flush=True)

    model.features.requires_grad_(False)
    if args.unfreeze_last:
        for block in list(model.features.children())[-args.unfreeze_last:]:
            block.requires_grad_(True)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    train_sampler = None
    if args.balance_classes:
        counts = Counter(label for _, label in splits["train"])
        weights = torch.tensor([1.0 / counts[label] for _, label in splits["train"]],
                               dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights,
                                              args.train_samples_per_epoch or len(weights),
                                              replacement=True,
                                              generator=torch.Generator().manual_seed(args.seed))
    loaders = {
        name: DataLoader(ImagePaths(items, training=name == "train"),
                         batch_size=args.batch_size,
                         shuffle=name == "train" and train_sampler is None,
                         sampler=train_sampler if name == "train" else None,
                         num_workers=0)
        for name, items in splits.items()
    }
    best_score = -1.0
    checkpoint_path = args.output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen batch-normalization statistics fixed to their pretrained values.
        model.features.eval()
        if args.unfreeze_last:
            for block in list(model.features.children())[-args.unfreeze_last:]:
                block.train()
        running_loss = 0.0
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        validation = evaluate(model, loaders["validation"], loss_function, device, len(classes))
        record = {"epoch": epoch, "train_loss": running_loss /
                  (args.train_samples_per_epoch or len(splits["train"])),
                  "validation_loss": validation["loss"],
                  "validation_accuracy": validation["accuracy"],
                  "validation_macro_recall": validation["macro_recall"]}
        history.append(record)
        print(f"epoch {epoch}/{args.epochs}: train_loss={record['train_loss']:.3f} "
              f"val_loss={validation['loss']:.3f} val_accuracy={validation['accuracy']:.3f} "
              f"val_macro_recall={validation['macro_recall']:.3f}",
              flush=True)
        if validation[args.selection_metric] > best_score:
            best_score = validation[args.selection_metric]
            weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"state_dict": weights, "classes": classes,
                        "architecture": "mobilenet_v3_small", "image_size": 224,
                        "source": str(args.data_dir)}, checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True)["state_dict"])
    result = {
        "source": str(args.data_dir),
        "classes": classes,
        "seed": args.seed,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate,
                     "max_per_class": args.max_per_class,
                     "start_from": str(args.start_from) if args.start_from else None,
                     "unfreeze_last": args.unfreeze_last,
                     "balance_classes": args.balance_classes,
                     "train_samples_per_epoch": args.train_samples_per_epoch,
                     "selection_metric": args.selection_metric},
        "history": history,
        "validation": evaluate(model, loaders["validation"], loss_function, device, len(classes)),
        "test": evaluate(model, loaders["test"], loss_function, device, len(classes)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {checkpoint_path}", flush=True)
    print(f"Held-out test accuracy: {result['test']['accuracy']:.3f} "
          f"({result['test']['count']} images)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
