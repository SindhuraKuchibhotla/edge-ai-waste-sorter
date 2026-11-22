"""Export the trained material classifier for the standalone UNO Q app."""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from .model import DEFAULT_CHECKPOINT, IMAGE_SIZE, load_checkpoint


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "uno_q_material_logger" / "model"


def export(checkpoint: Path, output_dir: Path) -> tuple[Path, Path]:
    model, labels = load_checkpoint(checkpoint, torch.device("cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "trashnet-mobilenetv3-small.onnx"
    labels_path = output_dir / "labels.json"
    sample = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        str(model_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(model_path))
    metadata = {
        "labels": labels,
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    }
    labels_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    model_path, labels_path = export(args.checkpoint, args.output_dir)
    print(f"ONNX model: {model_path}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
