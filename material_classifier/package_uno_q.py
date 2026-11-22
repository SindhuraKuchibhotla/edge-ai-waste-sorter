"""Build an importable App Lab archive for the UNO Q material OLED app."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


APP_ROOT = Path(__file__).resolve().parent.parent / "uno_q_material_logger"
DEFAULT_ARCHIVE = APP_ROOT.parent / "dist" / "material-classifier-oled-uno-q.zip"
SKETCH_FILES = tuple(
    str(path.relative_to(APP_ROOT.parent))
    for path in sorted((APP_ROOT.parent / "sketch").glob("*.ino"))
)
LIBRARY_FILES = tuple(
    str(path.relative_to(APP_ROOT.parent))
    for path in sorted((APP_ROOT.parent / "sketch" / "libraries").glob("*/*"))
    if path.is_file()
)
APP_FILES = (
    "app.yaml",
    "config.json",
    "README.md",
    "model/labels.json",
    "model/trashnet-mobilenetv3-small.onnx",
    "python/__init__.py",
    "python/display.py",
    "python/main.py",
    "python/material_logger.py",
    "python/requirements.txt",
    "sketch/sketch.yaml",
) + SKETCH_FILES + LIBRARY_FILES


def package(output: Path) -> Path:
    def source(name: str) -> Path:
        return APP_ROOT.parent / name if name.startswith("sketch/") else APP_ROOT / name

    missing = [name for name in APP_FILES if not source(name).is_file()]
    if missing:
        raise FileNotFoundError(f"App package is missing: {', '.join(missing)}")
    if output.resolve().is_relative_to(APP_ROOT):
        raise ValueError("write the archive outside the app directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in APP_FILES:
            archive.write(source(name), name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    print(package(args.output))


if __name__ == "__main__":
    main()
